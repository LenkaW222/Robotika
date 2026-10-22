#
import numpy as np
from ctu_mitsubishi import Rv6s

robot = Rv6s(debug=False)
robot.initialize()

q = robot.get_q()

position = robot.fk(q)
print(position)

pos = position @ np.array([1, 1, 1, 1])
print(pos)

q[0] = np.deg2rad(0)

robot.move_to_q(q)
robot.wait_for_motion_stop()

position = robot.fk(q)
pos = position @ np.array([1, 1, 1, 1])
print(pos)

robot.stop_robot()
robot.close_connection()