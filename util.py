from pydrake.all import (
    LeafSystem,
    BasicVector,
    MultibodyPlant,
    RigidBody,
    Context
)
import numpy as np
import matplotlib.pyplot as plt

class BodyStateExtractor(LeafSystem):
    def __init__(self, plant: MultibodyPlant, body: RigidBody):
        super().__init__()
        self.plant: MultibodyPlant = plant
        self.body_index = body.index()

        self.pose_input = self.DeclareAbstractInputPort("body_poses", plant.get_body_poses_output_port().Allocate())
        self.velocity_input = self.DeclareAbstractInputPort("body_velocities", plant.get_body_spatial_velocities_output_port().Allocate())

        self.output = self.DeclareVectorOutputPort("info", BasicVector(6), self.CalcOutput)

    def CalcOutput(self, context: Context, output: BasicVector):
        poses = self.EvalAbstractInput(context, 0).get_value()
        velocities = self.EvalAbstractInput(context, 1).get_value()

        p_WB = poses[self.body_index].translation()
        v_WB = velocities[self.body_index].translational()

        output.SetFromVector(np.concatenate([p_WB, v_WB]))


def plot_ee_traj(traj):
    T = traj.end_time()
    ts = np.linspace(0.0, T, 200)
    
    # Evaluate position & velocity
    pos = np.array([traj.value(t).flatten() for t in ts])
    vel = np.array([traj.derivative(1).value(t).flatten() for t in ts])

    # ---- Plot XY Trajectory ----
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

    # ---- Plot Velocity vs Time ----
    plt.figure()
    plt.plot(ts, vel[:, 0], label="vx")
    plt.plot(ts, vel[:, 1], label="vy")
    plt.plot(ts, np.linalg.norm(vel, axis=1), linestyle="--", label="|v|")
    plt.title("End Effector Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    plt.show()