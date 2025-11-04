from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    Context,
    BasicVector,
    JacobianWrtVariable
)
from constants import iiwa_q0
import numpy as np
from numpy.linalg import inv, pinv

class HFPController(LeafSystem):
    '''
    Hybrid Force Position Controller
    - Control motion in the xy-plane while exterting a force in the z-direction
    '''
    def __init__(self, plant: MultibodyPlant, f_z_des: float):
        super().__init__()
        self.f_des = np.array([0, 0, f_z_des])
        self.iiwa = plant.GetModelInstanceByName('iiwa7')
        self.gripper_body = plant.GetBodyByName('body')
        self.world_frame = plant.world_frame()
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()
        # self.iiwa_indices = [int(i) for i in self.plant.GetJointIndices(self.iiwa)]
        self.iiwa_indices = [i for i in range(1,8)]

        self.Kp = 300
        self.Kd = 15

        self.Kp_null = 100
        self.Kd_null = 15

        self.state_input = self.DeclareVectorInputPort('state', 14)
        self.traj_input = self.DeclareVectorInputPort('traj_input', 6)
        self.output_port = self.DeclareVectorOutputPort('torque', 7, self.calc_torque)

    def calc_torque(self, context: Context, output: BasicVector):
        state = self.state_input.Eval(context)
        pv = self.traj_input.Eval(context)
        q_meas = state[:7]
        v_meas = state[7:]
        path_p = pv[:3]
        path_pdot = pv[3:]

        # Calc Jacobian
        self.plant.SetPositions(self.plant_context, self.iiwa, q_meas)
        self.plant.SetVelocities(self.plant_context, self.iiwa, v_meas)
        J = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.gripper_body.body_frame(),
            [0,0,0],
            self.world_frame,
            self.world_frame
        )[:, self.iiwa_indices]

        # Calc gravity at EE
        M = self.plant.CalcMassMatrix(self.plant_context)[self.iiwa_indices, :][:, self.iiwa_indices]
        M_E = inv(J @ inv(M) @ J.T)
        tau_g = self.plant.CalcBiasTerm(self.plant_context)[self.iiwa_indices]
        f_g = M_E @ J @ inv(M) @ tau_g
        f_g = f_g[:3]

        # Get current p, pdot
        p_WG = self.gripper_body.EvalPoseInWorld(self.plant_context).translation()
        v_WG = self.gripper_body.EvalSpatialVelocityInWorld(self.plant_context).translational()

        # Calc input force at EE
        pd_xy = self.Kp*(path_p - p_WG) + self.Kd*(path_pdot - v_WG)
        pd_xy[2] = 0 # no explicit position velocity control in the z direction
        f_u = self.f_des - f_g + pd_xy
        f_u = np.concatenate((f_u, np.zeros(3)))
        
        # Calc torque with null space terms
        P = np.eye(7) - (J.T @ pinv(J @ J.T) @ J)
        u = J.T @ f_u + P @ (self.Kp_null*(iiwa_q0 - q_meas) - self.Kd_null*(v_meas))

        output.SetFromVector(u)
