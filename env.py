import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Diagram,
    DiagramBuilder,
    MeshcatVisualizer,
    ModelInstanceIndex,
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
    RotationMatrix,
    UnitInertia,
    SpatialInertia,
)

# TODO wtf
def add_table(
    plant: MultibodyPlant,
    dims: tuple[float, float, float],
    mu_static: float,
    mu_dynamic: float,
    color=[0.7, 0.5, 0.3, 1.0]
):
    model = plant.AddModelInstance('table_model')
    body = plant.AddRigidBody('table_body', model)
    shape = Box(*dims)
    pose = RigidTransform([0, 0, -dims[2]/2.0])
    plant.RegisterVisualGeometry(body, pose, shape, 'table_visual', color)
    plant.RegisterCollisionGeometry(
        body,
        pose,
        shape,
        "table_collision",
        CoulombFriction(static_friction=mu_static, dynamic_friction=mu_dynamic)
    )
    plant.WeldFrames(plant.world_frame(), body.body_frame())

    return body, model


def add_puck(
    plant: MultibodyPlant,
    dims: tuple[float, float],
    mu_static: float,
    mu_dynamic: float,
    pose: RigidTransform,
    mass: float,
    color=[0.0, 1.0, 0.0, 1.0]
):
    model = plant.AddModelInstance('puck_model')

    innertia = UnitInertia.SolidCylinder(radius=dims[0], length=dims[1], unit_vector=[0,0,1])
    spatial_inertia = SpatialInertia(mass=mass, p_PScm_E=[0, 0, 0], G_SP_E=innertia)

    body = plant.AddRigidBody('puck_body', model, spatial_inertia)
    shape = Cylinder(*dims)
    plant.RegisterVisualGeometry(body, pose, shape, 'puck_visual', color)
    plant.RegisterCollisionGeometry(
        body,
        pose,
        shape,
        "puck_collision",
        CoulombFriction(static_friction=mu_static, dynamic_friction=mu_dynamic)
    )

    return body, model


def make_system_diagram(
    meshcat: Meshcat,
    table_mu_static=0.9,
    table_mu_dynamic=0.5,
    table_dims=(10.0, 10.0, 0.4),
    puck_mu_static=0.9,
    puck_mu_dynamic=0.5,
    puck_dims=(0.053975, 0.0254),
    puck_mass=0.340,
    puck_pose=RigidTransform(RotationMatrix().MakeXRotation(0.0), [0,0,2.0])
) -> tuple[Diagram, MultibodyPlant, ModelInstanceIndex]:
    builder = DiagramBuilder()

    # make plant and scene_graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    plant: MultibodyPlant = plant
    scene_graph: SceneGraph = scene_graph

    # add and configure iiwa
    parser = Parser(plant, scene_graph)
    iiwa = parser.AddModelsFromUrl('package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf')[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('iiwa_link_0'))

    # add table surface
    table_body, table_model = add_table(plant, table_dims, table_mu_static, table_mu_dynamic)

    # add puck
    puck_body, puck_model = add_puck(plant, puck_dims, puck_mu_static, puck_mu_dynamic, puck_pose, puck_mass)

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    plant.Finalize()
    diagram = builder.Build()

    return diagram, plant, iiwa


class Sim():
    def __init__(self, diagram: Diagram, plant: MultibodyPlant, iiwa: ModelInstanceIndex, meshcat: Meshcat):
        self.diagram = diagram
        self.plant = plant
        self.iiwa = iiwa
        self.meshcat = meshcat

    def sim_passive(self, q0=np.zeros(7), sim_time=5.0):
        diagram_context = self.diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(diagram_context)

        self.plant.SetPositions(plant_context, self.iiwa, q0)
        
        self.plant.get_actuation_input_port().FixValue(plant_context, q0)

        sim = Simulator(diagram, diagram_context)

        sim.set_target_realtime_rate(1.0)

        meshcat.StartRecording()
        sim.AdvanceTo(sim_time)
        meshcat.StopRecording()
        meshcat.PublishRecording()        



if __name__ == '__main__':
    meshcat: Meshcat = StartMeshcat()
    diagram, plant, iiwa = make_system_diagram(meshcat)
    sim = Sim(diagram, plant, iiwa, meshcat)
    sim.sim_passive()

    while True:
        pass