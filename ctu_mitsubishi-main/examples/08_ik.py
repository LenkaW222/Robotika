import numpy as np
from ctu_mitsubishi import Rv6s

robot = Rv6s(debug=False)
robot.initialize()

q = robot.get_q()
print(q)
position = robot.fk(q)
print(position)
inverse = robot.ik(position)
print(inverse)

robot.stop_robot()
robot.close_connection()