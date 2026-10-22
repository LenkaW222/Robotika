#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-12-5
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
from numpy.typing import ArrayLike


def circle_circle_intersection(
    c0: ArrayLike, r0: float, c1: ArrayLike, r1: float
) -> list[np.ndarray]:
    """Computes intersection of the circles defined by center c_i and radius r_i.
    Returns empty array if there is no solution, two solutions otherwise. If there
    are infinite number of solutions, select two (almost) randomly.
    """
    c0 = np.asarray(c0)
    c1 = np.asarray(c1)
    d = np.linalg.norm(c1 - c0)
    if r0 + r1 < d < np.abs(r0 - r1) or d > r0 + r1:
        return []

    if np.isclose(d, 0) and np.isclose(r0, r1):
        return [
            c0 + [r0 * np.cos(a), r0 * np.sin(a)]
            for a in np.random.uniform(-np.pi, np.pi, 2)
        ]
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    assert r0**2 - a**2 >= 0, "Invalid circle_circle_intersection"

    h = np.sqrt(r0**2 - a**2)
    x0, y0 = c0
    x1, y1 = c1
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d
    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return [np.array([x3, y3]), np.array([x4, y4])]
