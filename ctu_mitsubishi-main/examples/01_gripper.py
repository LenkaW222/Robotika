#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-12-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from time import sleep

from ctu_mitsubishi import Rv6sGripper

gripper = Rv6sGripper()

for _ in range(5):
    gripper.open()
    sleep(1.5)
    gripper.close()
    sleep(1.5)

gripper.disconnect()
