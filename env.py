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
    ProximityProperties,
    ContactVisualizer,
    ContactVisualizerParams,
    RollPitchYaw,
    Sphere,
    Rgba,
    RigidBody,
    ModelInstanceIndex,
    PiecewisePolynomial,
    TrajectorySource,
    ConstantVectorSource,
    AddFrameTriadIllustration,
    RotationMatrix,
    SpatialVelocity
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
    table_x_offset
)
from controllers import HFPController, IK


def add_table(
    plant: MultibodyPlant,
    dims: tuple[float, float, float],
    mu_static: float,
    mu_dynamic: float,
    color=[0.7, 0.5, 0.3, 1.0]
) -> tuple[RigidBody, ModelInstanceIndex]:
    model = plant.AddModelInstance('table_model')
    body = plant.AddRigidBody('table_body', model)
    shape = Box(*dims)
    pose = RigidTransform([dims[0]/2.0 + table_x_offset, 0, -dims[2]/2.0])

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    AddRigidHydroelasticProperties(0.05, contact_properties)

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
    color=[0.0, 1.0, 0.0, 1.0]
) -> tuple[RigidBody, ModelInstanceIndex]:
    model = plant.AddModelInstance('puck_model')

    innertia = UnitInertia.SolidCylinder(radius=dims[0], length=dims[1], unit_vector=[0,0,1])
    spatial_inertia = SpatialInertia(mass=mass, p_PScm_E=[0, 0, 0], G_SP_E=innertia)

    body = plant.AddRigidBody('puck_body', model, spatial_inertia)
    shape = Cylinder(*dims)

    contact_properties = ProximityProperties()
    contact_properties.AddProperty('material', 'coulomb_friction', CoulombFriction(mu_static, mu_dynamic))
    AddRigidHydroelasticProperties(0.05, contact_properties)

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
        visualize_contact=True
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
        _, self.table = add_table(self.plant, table_dims, table_mu_static, table_mu_dynamic)

        # add puck
        self.puck_body, self.puck = add_puck(self.plant, puck_dims, puck_mu_static, puck_mu_dynamic, puck_mass)

        # add wsg
        self.gripper = parser.AddModelsFromUrl("package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf")[0]
        X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, 0), [0, 0, 0.09])
        self.plant.WeldFrames(self.plant.GetFrameByName('iiwa_link_7', self.iiwa), self.plant.GetFrameByName('body', self.gripper), X_7G)

        # contact
        self.plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
        self.plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

        self.plant.Finalize()

        if visualize_contact:
            ContactVisualizer.AddToBuilder(
                self.builder,
                self.plant,
                meshcat,
                ContactVisualizerParams()
            )

        # calc starting q given X_PuckG_init (in constants)
        self.q0 = IK(self.plant, X_WPuck_init@X_PuckG_init)

        # Add some visualization things
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
        meshcat.SetTransform('red_line', RigidTransform(R, [table_x_offset+1, 0, 0]))

    def test(self, sim_time=15.0):
        dummy_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(dummy_context, self.iiwa, iiwa_q0)

        # add controller
        controller = HFPController(self.plant, f_z_des=0.0)
        self.builder.AddNamedSystem('controller', controller)

        # trajectroy tracking test NOTE for these tests make sure the
        # selection matrices are set to position only
        # p_start = self.plant.GetBodyByName('body').EvalPoseInWorld(dummy_context).translation()
        # points = [
        #     list(p_start),
        #     [0.3, 0.2, 0.5],
        #     [0.3, 0, 0.2],
        #     [0, 0.2, 0.2]
        # ]
        # t_knots = [0.0, 5.0, 10.0, 15.0]
        # xy_knots = np.array(points).T
        # traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, xy_knots)
        # traj_source = TrajectorySource(traj, output_derivative_order=1)
        # self.builder.AddSystem(traj_source)
        # self.builder.Connect(traj_source.get_output_port(0), controller.traj_input)
        # for i, point in enumerate(points):
        #     self.meshcat.SetObject(f'{i}', Sphere(0.02), rgba=Rgba(0, 1, 0, 1))
        #     self.meshcat.SetTransform(f'{i}', RigidTransform(point))

        # position test NOTE selection matrices
        p_stationary = [0.5, 0.2, 0.3, 0, 0, 0]
        const_source = ConstantVectorSource(p_stationary)
        self.builder.AddSystem(const_source)
        self.builder.Connect(const_source.get_output_port(0), controller.traj_input)
        self.meshcat.SetObject(f'point', Sphere(0.02), rgba=Rgba(0, 1, 0, 1))
        self.meshcat.SetTransform(f'point', RigidTransform(p_stationary[:3]))

        # connect iiwa to controller
        self.builder.Connect(self.plant.get_state_output_port(self.iiwa), controller.state_input)
        self.builder.Connect(controller.output_port, self.plant.get_actuation_input_port(self.iiwa))

        # sim
        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)

        self.plant.SetPositions(plant_context, self.iiwa, iiwa_q0)
        
        puck_body = self.plant.GetBodyByName('puck_body')
        self.plant.SetFreeBodyPose(plant_context, puck_body, RigidTransform([3, 0, 0.2]))

        sim = Simulator(diagram, diagram_context)

        sim.set_target_realtime_rate(1.0)

        meshcat.StartRecording()
        sim.AdvanceTo(sim_time)
        meshcat.StopRecording()
        meshcat.PublishRecording()

    def test_basic(self):
        diagram = self.builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)
        
        self.plant.SetFreeBodyPose(plant_context, self.puck_body, X_WPuck_init)
        self.plant.SetFreeBodySpatialVelocity(self.puck_body, SpatialVelocity([0,0,0,1,0,0]), plant_context)
        self.plant.SetPositions(plant_context, self.iiwa, self.q0)

        sim = Simulator(diagram, diagram_context)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(5.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()

if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()
    env = Env(meshcat)
    env.test_basic()

    while True:
        pass