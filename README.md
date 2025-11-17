# shuffleboard_bot

Trajectory:

We want to keep our force constant to properly press on the puck to reach exact velocity and position of the end effector. The only thing we will calcuate is end effector trajectory, direction and velocity. We will keep a constant puck start position and randomly generate end goal positions at the end of the table. Once we have an end goal position for the puck, we will calculate the end goal position of the end effector which will be colinear to the start and end position of the puck. We will also calculate the velocity of the end effector in order to reach the desired velocity at the end of the end effector trajectory. 

Calculations:
We will use the formula Vf^2 = Vo^2 + 2a*(X - Xo)

We know the final velocity will be 0 since we want it to be stopped at the end goal
We know a, as the frictional force calculated by (- mu * mass of puck * g)
We know X and X0 which are the end goal position of the puck and start position of the puck, respectively

We want to find Vo as the velocity we want to achieve when we release the puck.

We can then derive:

Vo = sqrt(2*(-mu * mass of puck * g)*(X))
