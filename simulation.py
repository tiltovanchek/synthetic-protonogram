"""Пролет протонов через поля мишени и перенос на экран."""

import numpy as np

from fields import compute_fields
from pusher import boris_step


def track(r, v, target, dt, start_y, exit_y, q_over_m, max_steps):
    """Ведет протоны, пока все не покинут слой start_y <= y < exit_y.

    Протоны, вылетевшие из слоя вперед или назад, дальше не двигаются.
    """
    r, v = r.copy(), v.copy()
    for _ in range(max_steps):
        active = (r[:, 1] >= start_y) & (r[:, 1] < exit_y)
        if not active.any():
            break
        E, B = compute_fields(r[active], target)
        r[active], v[active] = boris_step(r[active], v[active], E, B, dt, q_over_m)
    else:
        print(f"Достигнут предел шагов, не вылетели: {np.sum(active)}")
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
