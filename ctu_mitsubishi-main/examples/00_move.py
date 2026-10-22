#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-11-28
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
from ctu_mitsubishi import Rv6s

robot = Rv6s(debug=False)
robot.initialize()

q = robot.get_q()
print(q)
q[5] = np.deg2rad(0)
robot.move_to_q(q)
robot.wait_for_motion_stop()
print(q)
# for i in range(6):
#     q[i] += np.deg2rad(10)
#     robot.move_to_q(q)
#     robot.wait_for_motion_stop()
#     print(f"Moved to {np.rad2deg(q).round()} degrees")
#     q[i] += np.deg2rad(-10)
#     robot.move_to_q(q)
#     robot.wait_for_motion_stop()
#     print(f"Moved to {np.rad2deg(q).round()} degrees")

robot.stop_robot()
robot.close_connection()
