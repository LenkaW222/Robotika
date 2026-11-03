import numpy as np
from ctu_mitsubishi import Rv6s

robot = Rv6s(debug=False)
robot.initialize()


q0 = robot.get_q()
print("Current q:", np.rad2deg(q0).round())

pose = robot.fk(q0)
print("Current pose:")
print(pose.round(3))


robot.stop_robot()
robot.close_connection()