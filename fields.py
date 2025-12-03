"""Электрическое и магнитное поле мишени."""

import numpy as np

from config import K_COULOMB, MU0_4PI


def compute_fields(r, target):
    """Поля E и B в точках r, (N, 3).

    Каждый отрезок мишени считается точечным зарядом (закон Кулона)
    и элементом тока (закон Био-Савара-Лапласа).
    """
    d = r[:, None, :] - target.mid[None, :, :]  # от отрезков к точкам, (N, M, 3)
    inv_r3 = np.sum(d * d, axis=2) ** -1.5  # 1 / |d|^3, (N, M)

    E = K_COULOMB * np.sum((target.dq * inv_r3)[..., None] * d, axis=1)
    dl_x_d = np.cross(target.dl[None, :, :], d)
    B = MU0_4PI * target.current * np.sum(inv_r3[..., None] * dl_x_d, axis=1)
    return E, B
