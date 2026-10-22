#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-11-28
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
from ctu_mitsubishi import Rv6s

robot = Rv6s(port=None)

q0 = robot.get_q()
print("Current q:", np.rad2deg(q0).round())

pose = robot.fk(q0)
print("Current pose:", pose.round(3))

offset_pose = np.eye(4)
offset_pose[2, 3] = 0.1
target_pose = pose @ offset_pose
print("Target pose:", target_pose.round(3))
qs = robot.ik(target_pose)

q = min(qs, key=lambda q: np.linalg.norm(q - q0))
print("Moving to q:", np.rad2deg(q).round())
robot.move_to_q(q)

robot.stop_robot()
robot.close_connection()
