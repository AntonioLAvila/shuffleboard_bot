from pydrake.all import RigidTransform, RotationMatrix
import numpy as np

puck_dims = (0.053975, 0.0254)
puck_mass=0.340
puck_mu_static=0.9
puck_mu_dynamic=0.5

table_dims = (3.6576, 0.6604, 0.1) # 144" x 26"
table_mu_static=0.9
table_mu_dynamic=0.5

table_x_offset = 0.3


iiwa_q0 = [0,0.1,0,-1.2,0,1.6,0]

X_WPuck_init = RigidTransform(RotationMatrix.Identity(), [0.4, 0.0, 0.02])
X_PuckG_init = RigidTransform(
    RotationMatrix.MakeXRotation(-np.pi/2) @ RotationMatrix.MakeYRotation(np.pi),
    [0, 0, 0.15]
)