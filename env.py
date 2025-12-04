import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
    SceneGraph,
    Meshcat,
    Box,
    RigidTransform,
    CoulombFriction,
    Cylinder,
    UnitInertia,
    SpatialInertia,
    DiscreteContactApproximation,
    ContactModel,
    AddRigidHydroelasticProperties,
    AddCompliantHydroelasticProperties,
    ProximityProperties,
    ContactVisualizer,
    ContactVisualizerParams,
    Sphere,
    Rgba,
    RigidBody,
    ModelInstanceIndex,
    AddFrameTriadIllustration,
    RotationMatrix,
    SpatialVelocity,
    TrajectorySource,
    LogVectorOutput
)
from constants import (
    table_dims,
    puck_dims,
    puck_mass,
    table_mu_dynamic,
    table_mu_static,
    puck_mu_dynamic,
    puck_mu_static,
    iiwa_q0,
    X_WPuck_init,
    X_PuckEE_init,
    ee_dims,
    ee_mass,
    ee_mu_dynamic,
    ee_mu_static,
    top_mass,
    top_mu_static,
    top_mu_dynamic,
    top_dims,
    x_limits,
    table_x_offset,
    gravity,
    model_mu
)
from util import BodyStateExtractor, plot_ee_traj
from controllers import HFPController, IK, make_EE_traj
import matplotlib.pyplot as plt


def add_table(
    plant: MultibodyPlant,
    dims: tuple[float, float, float],
    mu_static: float,
    mu_dynamic: float,
    color=[0.7, 0.5, 0.3, 1.0],
    contact_type='rigid'
) -> tuple[RigidBody, ModelInstanceIndex]:
    model = plant.AddModelInstance('table_model')
    body = plant.AddRigidBody('table_body', model)
    shape = Box(*dims)
    pose = RigidTransform([dims[0]/2.0 + table_x_offset, 0.2, -dims[2]/2.0])

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    if contact_type == 'rigid':
        AddRigidHydroelasticProperties(0.05, contact_properties)
    elif contact_type == 'compliant':
        AddCompliantHydroelasticProperties(0.05, 1e7, contact_properties)
    else:
        raise RuntimeError(f'Contact type {contact_type} not supported')

    plant.RegisterVisualGeometry(body, pose, shape, 'table_visual', color)
    plant.RegisterCollisionGeometry(
        body,
        pose,
        shape,
        "table_collision",
        contact_properties
    )
    plant.WeldFrames(plant.world_frame(), body.body_frame())

    return body, model

def add_cylinder(
    plant: MultibodyPlant,
    name: str,
    dims: tuple[float, float],
    mu_static: float,
    mu_dynamic: float,
    mass: float,
    color=[0.0, 1.0, 0.0, 1.0],
    contact_type='rigid',
    hydroelastic_modulus=1e7,
) -> tuple[RigidBody, ModelInstanceIndex]:
    model = plant.AddModelInstance(f'{name}_model')

    innertia = UnitInertia.SolidCylinder(radius=dims[0], length=dims[1], unit_vector=[0,0,1])
    spatial_inertia = SpatialInertia(mass=mass, p_PScm_E=[0, 0, 0], G_SP_E=innertia)

    body = plant.AddRigidBody(f'{name}_body', model, spatial_inertia)
    shape = Cylinder(*dims)

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    if contact_type == 'rigid':
        AddRigidHydroelasticProperties(0.05, contact_properties)
    elif contact_type == 'compliant':
        AddCompliantHydroelasticProperties(0.05, hydroelastic_modulus, contact_properties)
    else:
        raise RuntimeError(f'Contact type {contact_type} not supported')
    
    plant.RegisterVisualGeometry(body, RigidTransform(), shape, f'{name}_visual', color)
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        shape,
        f'{name}_collision',
        contact_properties
    )

    return body, model


class Env():
    def __init__(
        self,
        meshcat: Meshcat,
        time_step=1e-4,
        table_contact_tyype='rigid',
        puck_contact_type='compliant',
        ee_contact_type='rigid',
        debug_visualize=True
    ):  
        self.meshcat = meshcat
        self.builder = DiagramBuilder()

        # make plant and scene_graph
        plant, scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=time_step)
        self.plant: MultibodyPlant = plant
        self.scene_graph: SceneGraph = scene_graph

        # add meshcat
        MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat)

        # add and configure iiwa
        parser = Parser(self.plant, self.scene_graph)
        self.iiwa = parser.AddModelsFromUrl('package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf')[0]
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName('iiwa_link_0'), RigidTransform([0.7, -0.3, 0.0]) )

        # add table surface
        _, self.table = add_table(self.plant, table_dims, table_mu_static, table_mu_dynamic, contact_type=table_contact_tyype)

        # add puck
        self.puck_body, self.puck = add_cylinder(
            self.plant,
            'puck',
            puck_dims,
            puck_mu_static,
            puck_mu_dynamic,
            puck_mass,
            contact_type=puck_contact_type
        )
        self.top_body, self.top = add_cylinder(
            self.plant,
            'top',
            top_dims,
            top_mu_static,
            top_mu_dynamic,
            top_mass,
            color=[0.0, 0.0, 1.0, 1.0],
            contact_type=puck_contact_type
        )
        X_PTop = RigidTransform(RotationMatrix.Identity(), [0, 0, 1e-3/2 + puck_dims[1]/2])
        self.plant.WeldFrames(self.plant.GetFrameByName('puck_body', self.puck), self.plant.GetFrameByName('top_body', self.top), X_PTop)

        # add ee
        self.ee_body, self.ee = add_cylinder(
            self.plant,
            'ee',
            ee_dims,
            ee_mu_static,
            ee_mu_dynamic,
            ee_mass,
            color=[0.5, 0.5, 0.5, 1.0],
            contact_type=ee_contact_type
        )
        X_7EE = RigidTransform(RotationMatrix.Identity(), [0, 0, ee_dims[1]])
        self.plant.WeldFrames(self.plant.GetFrameByName('iiwa_link_7', self.iiwa), self.plant.GetFrameByName('ee_body', self.ee), X_7EE)

        # contact
        self.plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
        self.plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

        self.plant.Finalize()

        # Add some visualization things
        if debug_visualize:
            ContactVisualizer.AddToBuilder(
                self.builder,
                self.plant,
                meshcat,
                ContactVisualizerParams()
            )
            AddFrameTriadIllustration(
                scene_graph=self.scene_graph,
                body=self.plant.GetBodyByName('ee_body'),
                length=0.1
            )
            AddFrameTriadIllustration(
                scene_graph=self.scene_graph,
                body=self.puck_body,
                length=0.1
            )
            meshcat.SetObject('red_line', Cylinder(0.005, 2), rgba=Rgba(1, 0, 0, 1))
            R = RotationMatrix.MakeYRotation(np.pi/2) @ RotationMatrix.MakeXRotation(np.pi/2)
            meshcat.SetTransform('red_line', RigidTransform(R, [1.0 + table_x_offset, 0, 0]))

        # calc starting q given X_PuckEE_init (in constants)
        self.q0 = IK(self.plant, X_WPuck_init@X_PuckEE_init)

    def test_friction(self):
        '''
        Slides the puck on the table by setting an initial velocity 1m/s in x
        '''
        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)

        p_initial = X_WPuck_init.translation()[:2]
        p_final = np.array([2.0, 0.2])
        self.meshcat.SetObject('target', Sphere(0.01), rgba=Rgba(0,1,0,1))
        self.meshcat.SetTransform('target', RigidTransform(np.concatenate([p_final, [0]])))

        # calc p_release
        p_release = np.array([max(p_initial[0], x_limits[1]), p_initial[1]])
        # calc v_release
        d = p_final - p_release
        length = np.linalg.norm(d)
        v_release =  (d/length) * np.sqrt(2 * model_mu * gravity * length)
        self.plant.SetFreeBodyPose(plant_context, self.puck_body, RigidTransform([p_release[0], p_release[1], puck_dims[1]/2+1e-3]))
        self.plant.SetFreeBodySpatialVelocity(self.puck_body, SpatialVelocity(np.concatenate([[0,0,0], v_release, [0]])), plant_context)
        self.plant.SetPositions(plant_context, self.iiwa, iiwa_q0)

        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(5.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()
    
    def test_reach(self):
        '''
        change X_WP
        '''
        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)

        X_WP = RigidTransform(RotationMatrix.Identity(), [0.38, 0.0, puck_dims[1]/2+1e-3]) # change this
        q0 = IK(self.plant, X_WP@X_PuckEE_init)
        self.plant.SetPositions(plant_context, self.iiwa, q0)
        self.plant.SetFreeBodyPose(plant_context, self.puck_body, X_WP)

        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(1.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()
    
    def run_push(self):
        # set target
        # target = np.array([2.0, 0.2])
        target = np.array([3.0, 0.0])
        self.meshcat.SetObject('target', Sphere(0.01), rgba=Rgba(0,1,0,1))
        self.meshcat.SetTransform('target', RigidTransform(np.concatenate([target, [0]])))

        # Make controller and make trajectories
        controller = HFPController(self.plant, S_force=np.diag([0,0,0,1,1,1]))
        # controller = HFPController(self.plant)
        EE_spatial_traj, EE_fz_traj = make_EE_traj(X_WPuck_init.translation()[:2], target)
        EE_pos_source = TrajectorySource(EE_spatial_traj)
        EE_vel_source = TrajectorySource(EE_spatial_traj.derivative(1))
        EE_fz_source = TrajectorySource(EE_fz_traj)

        # Add them to the builder
        self.builder.AddNamedSystem('hfp_controller', controller)
        self.builder.AddNamedSystem('ee_pos', EE_pos_source)
        self.builder.AddNamedSystem('ee_vel', EE_vel_source)
        self.builder.AddNamedSystem('ee_fz', EE_fz_source)

        # Connect everything
        self.builder.Connect(self.plant.get_state_output_port(self.iiwa), controller.state_input)
        self.builder.Connect(EE_pos_source.get_output_port(), controller.traj_pos_input)
        self.builder.Connect(EE_vel_source.get_output_port(), controller.traj_vel_input)
        self.builder.Connect(EE_fz_source.get_output_port(), controller.force_input)
        self.builder.Connect(controller.output_port, self.plant.get_actuation_input_port(self.iiwa))

        # EE traj log TODO fix this
        extractor = BodyStateExtractor(self.plant, self.ee_body)
        self.builder.AddNamedSystem('extractor', extractor)
        self.builder.Connect(self.plant.get_body_poses_output_port(), extractor.pose_input)
        self.builder.Connect(self.plant.get_body_spatial_velocities_output_port(), extractor.velocity_input)
        logger = LogVectorOutput(extractor.output, self.builder)
        logger.set_name('ee_log')

        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)

        self.plant.SetFreeBodyPose(plant_context, self.puck_body, X_WPuck_init)
        self.plant.SetPositions(plant_context, self.iiwa, self.q0)

        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(7.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()

        # TODO trim these to the same times and compare agains the expected
        ee_log = logger.FindLog(diagram_context)
        times = ee_log.sample_times()
        data = ee_log.data()  # shape (N, samples)
        x  = data[0, :]
        y  = data[1, :]
        vx = data[3, :]
        vy = data[4, :]

        T = EE_spatial_traj.end_time()
        ts = np.linspace(0.0, T, 200)
        
        # Evaluate position & velocity
        pos = np.array([EE_spatial_traj.value(t).flatten() for t in ts])
        vel = np.array([EE_spatial_traj.derivative(1).value(t).flatten() for t in ts])

        # --------------------  Plot EE XY trajectory  --------------------
        plt.figure()
        plt.plot(pos[:, 0], pos[:, 1], label="EE path")
        # plt.scatter(p_initial[0], p_initial[1], color="green", label="Start")
        # plt.scatter(pos[-1, 0], pos[-1, 1], color="red", label="Release")
        plt.title("End Effector XY Trajectory")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        # plt.figure()
        plt.plot(x, y)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("End Effector XY Trajectory")
        plt.axis("equal")
        plt.grid(True)

        # --------------------  Plot XY velocities  --------------------
        plt.figure()
        plt.plot(ts, vel[:, 0], label="wantedvx")
        plt.plot(ts, vel[:, 1], label="wantedvy")
        plt.plot(ts, np.linalg.norm(vel, axis=1), linestyle="--", label="|wantedv|")
        plt.title("End Effector Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True)
        plt.legend()

        # plt.figure()
        plt.plot(times, vx, label="vx")
        plt.plot(times, vy, label="vy")
        plt.plot(times, np.sqrt(vx**2 + vy**2), label="|v|")
        plt.xlabel("time (s)")
        plt.ylabel("velocity (m/s)")
        plt.title("End Effector XY Velocities")
        plt.legend()
        plt.grid(True)

        plt.show()

        # plot_ee_traj(EE_spatial_traj)



if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()
    env = Env(meshcat)
    # env.test_reach()
    # env.test_friction()
    env.run_push()

    while True:
        pass