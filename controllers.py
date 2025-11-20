from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    Context,
    BasicVector,
    JacobianWrtVariable,
    RigidTransform,
    InverseKinematics,
    RotationMatrix,
    Solve,
    PiecewisePolynomial
)
from constants import (
    iiwa_q0,
    model_mu,
    gravity,
    press_force_mag,
    x_limits
)
import numpy as np
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt


class HFPController(LeafSystem):
    '''
    Hybrid Force Position Controller
    - Control motion in the xy-plane while exterting a force in the z-direction
    '''
    def __init__(self, plant: MultibodyPlant, S_pos=np.diag([1,1,1,1,1,0]), S_force=np.diag([0,0,0,0,0,1])):
        '''
        Default selection matrices choose position control in rotation xyz and position xy,
        and force control in z
        '''
        super().__init__()
        self.S_pos = S_pos
        self.S_force = S_force
        self.iiwa = plant.GetModelInstanceByName('iiwa7')
        self.gripper_body = plant.GetBodyByName('ee_body')
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()
        all_v = np.arange(self.plant.num_velocities())
        self.iiwa_indices = self.plant.GetVelocitiesFromArray(self.iiwa, all_v).astype(int)

        # gains
        # generaly set kd = 2*sqrt(kp)
        # TODO tune
        self.Kp = 1000
        self.Kd = 2*np.sqrt(self.Kp)

        self.Kp_tau = 10000
        self.Kd_tau = 2*np.sqrt(self.Kp_tau)

        # this ones probably fine?
        self.Kp_null = 100
        self.Kd_null = 60

        # io
        self.state_input = self.DeclareVectorInputPort('state', 14) # input is q, qdot
        self.traj_pos_input = self.DeclareVectorInputPort('traj_pos_input', 2) # input is EE p_xy
        self.traj_vel_input = self.DeclareVectorInputPort('traj_vel_input', 2) # input is EE pdot_xy
        self.force_input = self.DeclareVectorInputPort('f_z', 1) # force in the z direction
        self.output_port = self.DeclareVectorOutputPort('torque', 7, self.calc_torque) # output is 7 torques

    def calc_torque(self, context: Context, output: BasicVector):
        state = self.state_input.Eval(context)
        path_p = self.traj_pos_input.Eval(context)
        path_pdot = self.traj_vel_input.Eval(context)
        f_z = self.force_input.Eval(context)
        q_meas = state[:7]
        v_meas = state[7:]
        path_p = np.concatenate([path_p, [0]])
        path_pdot = np.concatenate([path_pdot, [0]])
        F_des = np.concatenate([np.zeros(5), f_z])

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
        # NOTE this should be done by CalcRelativeTransform but this is easier
        X_WG = self.gripper_body.EvalPoseInWorld(self.plant_context)
        V_WG = self.gripper_body.EvalSpatialVelocityInWorld(self.plant_context)

        p_WG, R_WG = X_WG.translation(), X_WG.rotation()
        v_WG, w_WG = V_WG.translational(), V_WG.rotational()

        # Calc input force at EE
        f_pv = self.Kp*(path_p - p_WG) + self.Kd*(path_pdot - v_WG)
        tau_pv = -self.Kd_tau*w_WG # this drives rotational position to not move from starting
        F_pv = np.concatenate((tau_pv, f_pv))
        F_u = (self.S_pos @ F_pv) + (self.S_force @ F_des) - F_g
        
        # Calc torque with null space terms
        P = np.eye(7) - pinv(J) @ J
        u = J.T @ F_u + P @ (self.Kp_null*(iiwa_q0 - q_meas) - self.Kd_null*(v_meas))

        output.SetFromVector(u)


def IK(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    tolerance=0.001,
    theta_bound=np.pi/180,
    n_tries=100
):
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName('ee_body')

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


def make_EE_traj(p_initial: np.ndarray, p_final: np.ndarray, time=0.8) -> tuple[PiecewisePolynomial, PiecewisePolynomial]:
    '''
    Return a trajectory for the end effector in the xy-plane and a force trajectory
    in the z direction.
    - The inputs are only in the xy-plane
    - use model_mu in constants to determine trajectory
    '''
    # calc p_release
    p_release = np.array([max(p_initial[0], x_limits[1]), p_initial[1]])

    # calc v_release
    d = p_final - p_release
    length = norm(d)
    v_release =  (d/length) * np.sqrt(2 * model_mu * gravity * length)

    # calc path
    breaks = [0.0, time]
    traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        breaks=breaks,
        samples=[p_initial.reshape(2,1), p_release.reshape(2,1)],
        sample_dot_at_start=np.zeros((2,1)),
        sample_dot_at_end=v_release.reshape(2,1)
    )
    force_traj = PiecewisePolynomial.ZeroOrderHold(
        breaks=breaks,
        samples=[np.array([[-press_force_mag]]), np.array([[-press_force_mag]])] # NOTE this
    )

    return traj, force_traj


if __name__ == "__main__":
    p_initial = np.array([0.4, 0.0])
    p_final = np.array([3.0, 0.5])
    traj, f_traj = make_EE_traj(p_initial, p_final)

    T = traj.end_time()
    ts = np.linspace(0.0, T, 200)

    # Evaluate force
    force = np.array([f_traj.value(t).flatten() for t in ts])
    
    # Evaluate position & velocity
    pos = np.array([traj.value(t).flatten() for t in ts])
    vel = np.array([traj.derivative(1).value(t).flatten() for t in ts])

    # ---- Plot XY Trajectory ----
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], label="EE path")
    plt.scatter(p_initial[0], p_initial[1], color="green", label="Start")
    plt.scatter(pos[-1, 0], pos[-1, 1], color="red", label="Release")
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
