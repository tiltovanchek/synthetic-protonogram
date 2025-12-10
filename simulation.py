"""Пролет протонов через поля мишени и перенос на экран."""

import numpy as np
from numba import njit, prange

from fields import fields_at
from pusher import boris_step


@njit(parallel=True, cache=True)
def _track_all(r0, v0, mid, dl, dq, current, dt, start_y, exit_y, q_over_m, max_steps):
    """Протоны не влияют друг на друга, поэтому каждый ведется
    целиком в своем потоке (prange), без синхронизации на шагах.
    """
    r = np.empty_like(r0)
    v = np.empty_like(v0)
    for i in prange(r0.shape[0]):
        p = (r0[i, 0], r0[i, 1], r0[i, 2])
        u = (v0[i, 0], v0[i, 1], v0[i, 2])
        for _ in range(max_steps):
            if p[1] < start_y or p[1] >= exit_y:
                break
            E, B = fields_at(p, mid, dl, dq, current)
            p, u = boris_step(p, u, E, B, dt, q_over_m)
        r[i, 0], r[i, 1], r[i, 2] = p
        v[i, 0], v[i, 1], v[i, 2] = u
    return r, v


def track(r, v, target, dt, start_y, exit_y, q_over_m, max_steps):
    """Ведет протоны, пока все не покинут слой start_y <= y < exit_y.

    Протоны, вылетевшие из слоя вперед или назад, дальше не двигаются.
    """
    r, v = _track_all(r, v, target.mid, target.dl, target.dq, target.current,
                      dt, start_y, exit_y, q_over_m, max_steps)
    stuck = np.sum((r[:, 1] >= start_y) & (r[:, 1] < exit_y))
    if stuck:
        print(f"Достигнут предел шагов, не вылетели: {stuck}")
    return r, v


def project_to_screen(r, v, screen_y):
    """Продолжает прямые траектории до экрана y = screen_y.

    Возвращает координаты x, z на экране.
    Протоны, летящие от экрана, отбрасываются.
    """
    ok = v[:, 1] > 0
    r, v = r[ok], v[ok]
    t = (screen_y - r[:, 1]) / v[:, 1]
    return r[:, 0] + v[:, 0] * t, r[:, 2] + v[:, 2] * t
