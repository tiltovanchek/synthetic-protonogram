"""Электрическое и магнитное поле мишени."""

import numpy as np
from numba import njit

from config import K_COULOMB, MU0_4PI
from vec3 import add, cross, scale


@njit(cache=True)
def fields_at(r, mid, dl, dq, current):
    """Поля E и B в точке r = (x, y, z).

    Каждый отрезок мишени считается точечным зарядом (закон Кулона)
    и элементом тока (закон Био-Савара-Лапласа).
    """
    E = (0.0, 0.0, 0.0)
    B = (0.0, 0.0, 0.0)
    for j in range(mid.shape[0]):
        d = (r[0] - mid[j, 0], r[1] - mid[j, 1], r[2] - mid[j, 2])
        d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        inv_r3 = 1.0 / (d2 * np.sqrt(d2))
        E = add(E, scale(d, dq[j] * inv_r3))
        B = add(B, scale(cross((dl[j, 0], dl[j, 1], dl[j, 2]), d), inv_r3))
    return scale(E, K_COULOMB), scale(B, MU0_4PI * current)
