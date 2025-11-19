from pydrake.all import RigidTransform, RotationMatrix
import numpy as np

puck_dims = (0.053975, 0.0254)
puck_mass=0.340
puck_mu_static=0.04
puck_mu_dynamic=0.03

table_dims = (3.6576, 0.6604, 0.1) # 144" x 26"
table_mu_static=0.05
table_mu_dynamic=0.03

# TODO come up with reasonable values
ee_dims = (0.01, 0.07)
ee_mass=0.015
ee_mu_static=0.9
ee_mu_dynamic=0.5

table_x_offset = 0.3
cutoff = 1.0
gravity = 9.80665

press_force_mag = 10.0 # TODO derive this

model_mu = (2 * puck_mu_dynamic * table_mu_dynamic) / (puck_mu_dynamic + table_mu_dynamic)

iiwa_q0 = [0,0.1,0,-1.2,0,1.6,0]

X_WPuck_init = RigidTransform(RotationMatrix.Identity(), [0.4, 0.0, puck_dims[1]/2+1e-3])
X_PuckEE_init = RigidTransform(
    RotationMatrix.MakeYRotation(np.pi),
    [0, 0, 0.05]
)