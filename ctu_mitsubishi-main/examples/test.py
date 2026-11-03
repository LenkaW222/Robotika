import numpy as np
from ctu_mitsubishi import Rv6s

robot = Rv6s(debug=False)
robot.initialize()


q = robot.get_q()
print("Current q:", np.rad2deg(q).round())

q[0] = np.deg2rad(5)
q[1] = np.deg2rad(0)
q[2] = np.deg2rad(90)
q[3] = np.deg2rad(0)
q[4] = np.deg2rad(90)
q[5] = np.deg2rad(180)

robot.move_to_q(q)
robot.wait_for_motion_stop()

pose = robot.fk(q)
print("Current pose:")
print(pose.round(3))

robot.stop_robot()
robot.close_connection()