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
    RollPitchYaw,
    Sphere,
    Rgba,
    RigidBody,
    ModelInstanceIndex,
    AddFrameTriadIllustration,
    RotationMatrix,
    SpatialVelocity,
    TrajectorySource,
    ConstantVectorSource
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
    X_PuckG_init,
    table_x_offset,
    cutoff
)
from controllers import HFPController, IK, make_EE_traj

# TODO what should the hydroelastic modulus be?

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
    pose = RigidTransform([dims[0]/2.0 + table_x_offset, 0, -dims[2]/2.0])

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    if contact_type == 'rigid':
        AddRigidHydroelasticProperties(0.05, contact_properties)
    elif contact_type == 'compliant':
        AddCompliantHydroelasticProperties(0.05, 100000, contact_properties)
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


def add_puck(
    plant: MultibodyPlant,
    dims: tuple[float, float],
    mu_static: float,
    mu_dynamic: float,
    mass: float,
    color=[0.0, 1.0, 0.0, 1.0],
    contact_type='rigid'
) -> tuple[RigidBody, ModelInstanceIndex]:
    model = plant.AddModelInstance('puck_model')

    innertia = UnitInertia.SolidCylinder(radius=dims[0], length=dims[1], unit_vector=[0,0,1])
    spatial_inertia = SpatialInertia(mass=mass, p_PScm_E=[0, 0, 0], G_SP_E=innertia)

    body = plant.AddRigidBody('puck_body', model, spatial_inertia)
    shape = Cylinder(*dims)

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    if contact_type == 'rigid':
        AddRigidHydroelasticProperties(0.05, contact_properties)
    elif contact_type == 'compliant':
        AddCompliantHydroelasticProperties(0.05, 100000, contact_properties)
    else:
        raise RuntimeError(f'Contact type {contact_type} not supported')

    plant.RegisterVisualGeometry(body, RigidTransform(), shape, 'puck_visual', color)
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        shape,
        "puck_collision",
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
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName('iiwa_link_0'))

        # add table surface
        _, self.table = add_table(self.plant, table_dims, table_mu_static, table_mu_dynamic, contact_type=table_contact_tyype)

        # add puck
        self.puck_body, self.puck = add_puck(self.plant, puck_dims, puck_mu_static, puck_mu_dynamic, puck_mass, contact_type=puck_contact_type)

        # add wsg
        self.gripper = parser.AddModelsFromUrl("package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf")[0]
        X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, 0), [0, 0, 0.09])
        self.plant.WeldFrames(self.plant.GetFrameByName('iiwa_link_7', self.iiwa), self.plant.GetFrameByName('body', self.gripper), X_7G)

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
                body=self.plant.GetBodyByName('body'),
                length=0.1
            )
            AddFrameTriadIllustration(
                scene_graph=self.scene_graph,
                body=self.puck_body,
                length=0.1
            )
            meshcat.SetObject('red_line', Cylinder(0.005, 2), rgba=Rgba(1, 0, 0, 1))
            R = RotationMatrix.MakeYRotation(np.pi/2) @ RotationMatrix.MakeXRotation(np.pi/2)
            meshcat.SetTransform('red_line', RigidTransform(R, [table_x_offset+cutoff, 0, 0]))

        # calc starting q given X_PuckG_init (in constants)
        # TODO doesnt work
        self.q0 = IK(self.plant, X_WPuck_init@X_PuckG_init)

        # Keep the fingers shut
        finger_l = self.plant.GetJointByName("left_finger_sliding_joint", self.gripper)
        finger_r = self.plant.GetJointByName("right_finger_sliding_joint", self.gripper)
        finger_l.set_default_positions([0.0])
        finger_r.set_default_positions([0.0])
        const_source = ConstantVectorSource([0.0, 0.0])
        self.builder.AddSystem(const_source)
        self.builder.Connect(const_source.get_output_port(), self.plant.get_actuation_input_port(self.gripper))

    def test_friction(self):
        '''
        Slides the puck on the table by setting an initial velocity 1m/s in x
        '''
        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)
        
        self.plant.SetFreeBodyPose(plant_context, self.puck_body, X_WPuck_init)
        self.plant.SetFreeBodySpatialVelocity(self.puck_body, SpatialVelocity([0,0,0,1,0,0]), plant_context)
        self.plant.SetPositions(plant_context, self.iiwa, iiwa_q0)

        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(5.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()
    
    def run_push(self):
        # Make controller and make trajectories
        controller = HFPController(self.plant)
        EE_spatial_traj, EE_fz_traj = make_EE_traj(X_WPuck_init.translation()[:2], np.array([2, 0.2]), time=2.0)
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

        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)

        self.plant.SetFreeBodyPose(plant_context, self.puck_body, X_WPuck_init)
        self.plant.SetFreeBodySpatialVelocity(self.puck_body, SpatialVelocity([0,0,0,0,0,0]), plant_context)
        self.plant.SetPositions(plant_context, self.iiwa, self.q0)

        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(10.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()

if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()
    env = Env(meshcat)
    # env.test_basic()
    env.run_push()

    while True:
        pass