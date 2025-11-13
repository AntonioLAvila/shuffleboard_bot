from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    Context,
    BasicVector,
    JacobianWrtVariable,
    KinematicTrajectoryOptimization,
    RigidTransform,
    InverseKinematics,
    RotationMatrix,
    Solve
)
from constants import iiwa_q0
import numpy as np
from numpy.linalg import inv, pinv


class HFPController(LeafSystem):
    '''
    Hybrid Force Position Controller
    - Control motion in the xy-plane while exterting a force in the z-direction
    '''
    def __init__(self, plant: MultibodyPlant, f_z_des: float, S_pos=np.diag([1,1,0,1,1,0]), S_force=np.diag([0,0,0,0,0,1])):
        '''
        Default selection matrices choose position control in rotation xy and position xy,
        and force control in z
        '''
        super().__init__()
        self.F_des = np.array([0, 0, 0, 0, 0, f_z_des])
        self.S_pos = S_pos
        self.S_force = S_force
        self.iiwa = plant.GetModelInstanceByName('iiwa7')
        self.gripper_body = plant.GetBodyByName('body')
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()
        
        all_v = np.arange(self.plant.num_velocities())
        self.iiwa_indices = self.plant.GetVelocitiesFromArray(self.iiwa, all_v).astype(int)

        self.Kp = 100
        self.Kd = 60

        self.Kp_tau = 100
        self.Kd_tau = 30

        self.Kp_null = 100
        self.Kd_null = 60

        self.state_input = self.DeclareVectorInputPort('state', 14) # input is q, qdot
        self.traj_input = self.DeclareVectorInputPort('traj_input', 6) # input is EE p, pdot
        self.output_port = self.DeclareVectorOutputPort('torque', 7, self.calc_torque) # output is torques

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
            self.plant.world_frame(),
            self.plant.world_frame()
        )[:, self.iiwa_indices]

        # Calc gravity at EE
        M = self.plant.CalcMassMatrix(self.plant_context)[self.iiwa_indices, :][:, self.iiwa_indices]
        M_E = inv(J @ inv(M) @ J.T)
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)[self.iiwa_indices]
        F_g = M_E @ J @ inv(M) @ tau_g

        # Get current p, pdot
        X_WG = self.gripper_body.EvalPoseInWorld(self.plant_context)
        V_WG = self.gripper_body.EvalSpatialVelocityInWorld(self.plant_context)

        p_WG, R_WG = X_WG.translation(), X_WG.rotation()
        v_WG, w_WG = V_WG.translational(), V_WG.rotational()

        # Calc input force at EE
        f_pv = self.Kp*(path_p - p_WG) + self.Kd*(path_pdot - v_WG)
        tau_pv = -self.Kd_tau*w_WG # NOTE this drives rotational position to 0 depending on S_pos
        F_pv = np.concatenate((tau_pv, f_pv))
        F_u = (self.S_pos @ F_pv) + (self.S_force @ self.F_des) - F_g
        
        # Calc torque with null space terms
        P = np.eye(7) - pinv(J) @ J
        u = J.T @ F_u + P @ (self.Kp_null*(iiwa_q0 - q_meas) - self.Kd_null*(v_meas))

        output.SetFromVector(u)


def IK(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    tolerance=0.001,
    theta_bound=1*np.pi/180,
    n_tries=100
):
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName('body')

    ik = InverseKinematics(plant)
    prog = ik.prog()
    q_vars = ik.q()[:7]
    
    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=X_WG_target.rotation(),
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix().Identity(),
        theta_bound=theta_bound
    )

    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=[0,0,0],
        frameA=world_frame,
        p_AQ_lower=X_WG_target.translation() - tolerance,
        p_AQ_upper=X_WG_target.translation() + tolerance
    )

    prog.AddQuadraticErrorCost(1.0, iiwa_q0, q_vars)

    lb, ub = plant.GetPositionLowerLimits()[:7], plant.GetPositionUpperLimits()[:7]
    for _ in range(n_tries):
        guess = lb + (ub - lb) * np.random.rand(7)
        prog.SetInitialGuess(q_vars, guess)
        result = Solve(prog)
        if result.is_success():
            return tuple(result.GetSolution(q_vars))
    raise RuntimeError('IK failed')


def make_EE_traj_xy():
    trajopt = KinematicTrajectoryOptimization(7, 15)
    prog = trajopt.prog()