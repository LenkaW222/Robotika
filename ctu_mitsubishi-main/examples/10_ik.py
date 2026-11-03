import numpy as np
from ctu_mitsubishi import Rv6s
from time import sleep

robot = Rv6s(debug=False)
robot.initialize()


position = np.array(
    [[1, 0, 0, 0.4],
     [0, -1, 0, 0],
     [0, 0, -1, 0.40],
     [0, 0, 0, 1]])

q0 = robot.get_q()
print(q0)
qs = robot.ik(position)

qnew = min(qs, key=lambda q: np.linalg.norm(q-q0))


print("q new: ", end=" ")
print(qnew)

steps = 5

for i in range(steps):
    qmove = q0 - (q0-qnew)*(i/(steps-1))
    robot.move_to_q(qmove)
    robot.wait_for_motion_stop()
    sleep(0.2)
    print("steps(" + str(i+1) + "/" + str(steps)+"), ", end = "")
    print(qmove)

robot.stop_robot()
robot.close_connection()