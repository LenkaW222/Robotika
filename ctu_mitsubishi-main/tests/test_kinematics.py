# !/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-10-31
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
import unittest
from ctu_mitsubishi import Rv6s
from ctu_mitsubishi.utils import circle_circle_intersection


class TestKinematics(unittest.TestCase):
    def test_fk0(self):
        q = np.deg2rad([0, 0, 90, 0, 0, 0])
        r = Rv6s(port=None)
        pose = r.fk(q)
        exp_z = 0.73
        exp_x = 0.485
        exp_y = 0
        theta = np.pi / 2
        exp_rot = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        np.testing.assert_allclose(pose[:3, 3], [exp_x, exp_y, exp_z], atol=1e-6)
        np.testing.assert_allclose(pose[:3, :3], exp_rot, atol=1e-6)

    def test_circ_circ_intersection(self):
        ints = circle_circle_intersection([0, 0], 1, [1, 0], 1)
        self.assertEqual(len(ints), 2)
        (ax, ay), (bx, by) = ints
        if by > ay:
            ay, by = by, ay
        self.assertAlmostEqual(ax, 0.5)
        self.assertAlmostEqual(ay, np.sqrt(1 - 0.5**2))
        self.assertAlmostEqual(bx, 0.5)
        self.assertAlmostEqual(by, -np.sqrt(1 - 0.5**2))

        ints = circle_circle_intersection([0, 0], 1, [2, 0], 1)
        self.assertEqual(len(ints), 2)
        (ax, ay), (bx, by) = ints
        self.assertAlmostEqual(ax, 1.0)
        self.assertAlmostEqual(ay, 0.0)
        self.assertAlmostEqual(bx, 1.0)
        self.assertAlmostEqual(by, 0.0)

        ints = circle_circle_intersection([0, 0], 1, [0, 0], 1)
        self.assertEqual(len(ints), 2)
        a, b = ints
        self.assertAlmostEqual(np.linalg.norm(a), 1.0)
        self.assertAlmostEqual(np.linalg.norm(b), 1.0)

        ints = circle_circle_intersection([0, 0], 1, [5, 0], 1)
        self.assertEqual(len(ints), 0)

    def test_ik_xyz(self):
        np.random.seed(0)
        r = Rv6s(port=None)

        def fk_5th_joint_pos(q):
            t = np.eye(4)
            t[2, 3] = -r.dh_d[-1]
            return (r.fk(q) @ t)[:3, 3]

        for _ in range(100):
            q = np.random.uniform(r.q_min, r.q_max)
            exp_pos = fk_5th_joint_pos(q)
            sols = r._ik_5th_joint_pos(exp_pos)
            self.assertTrue(any([np.allclose(q[:3], s, atol=1e-6) for s in sols]))
            for s in sols:
                q[:3] = s
                np.testing.assert_allclose(fk_5th_joint_pos(q), exp_pos, atol=1e-6)

    def test_ik(self):
        np.random.seed(0)
        r = Rv6s(port=None)
        for i in range(100):
            q = np.random.uniform(r.q_min, r.q_max)
            pose = r.fk(q)
            sols = r.ik(pose)
            self.assertTrue(any([np.allclose(q, s, atol=1e-6) for s in sols]))
            for c in sols:
                np.testing.assert_allclose(r.fk(c), pose, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
