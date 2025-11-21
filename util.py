from pydrake.all import (
    LeafSystem,
    BasicVector,
    MultibodyPlant,
    RigidBody
)
import numpy as np

class BodyStateReporter(LeafSystem):
    def __init__(self, plant: MultibodyPlant, body: RigidBody):
        super().__init__()
        self.plant: MultibodyPlant = plant
        self.body: RigidBody = body

        self.ouput_port = self.DeclareVectorOutputPort("state", BasicVector(6), self.CalcVelocity)

    def CalcVelocity(self, context, output):
        plant_context = self.plant.GetMyContextFromRoot(context)
        p_WB = self.body.EvalPoseInWorld(plant_context).translation()
        v_WB = self.body.EvalSpatialVelocityInWorld(plant_context).translational()
        output.SetFromVector(np.concatenate([p_WB, v_WB]))
