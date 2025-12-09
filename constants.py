from pydrake.all import RigidTransform, RotationMatrix
import numpy as np

table_dims = (3.6576, 0.6604, 0.1) # 144" x 26"
table_mu_static = 0.05
table_mu_dynamic = 0.02

puck_dims = (0.05, 0.02)
puck_mass = 0.340
puck_mu_static = 0.04
puck_mu_dynamic = 0.01
top_mass = 1e-3
top_mu_static = 0.9
top_mu_dynamic = 0.8
top_dims = (0.05, 1e-3)

ee_dims = (0.01, 0.07)
ee_mass = 0.015
ee_mu_static = 0.9
ee_mu_dynamic = 0.8

table_x_offset = 0.4
gravity = 9.80665

press_force_mag = 50.0

model_mu = (2 * puck_mu_dynamic * table_mu_dynamic) / (puck_mu_dynamic + table_mu_dynamic)

iiwa_q0 = [0,0.1,0,-1.2,0,1.6,0]

X_WPuck_init = RigidTransform(RotationMatrix.Identity(), [0.5, 0.25, puck_dims[1]/2+1e-3])
X_PuckEE_init = RigidTransform(
    RotationMatrix.MakeYRotation(np.pi),
    [0, 0, puck_dims[1]/2 + top_dims[1] + ee_dims[1]/2 + 2e-3]
)

x_limits = (0.2, 0.75) # meters 