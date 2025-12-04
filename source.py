"""Генерация протонов методом Монте-Карло."""

import numpy as np


def generate_protons(n, rng, energy_mev, theta_range, source_y, start_y, mass, mev):
    """Протоны из точечного источника на оси y, летящие вдоль +y.

    Возвращает координаты и скорости на плоскости y = start_y, (n, 3).
    Энергия и угол theta к оси y равномерны в заданных пределах,
    азимут fi равномерен от 0 до 2 pi.
    """
    energy = rng.uniform(*energy_mev, n) * mev
    speed = np.sqrt(2 * energy / mass)
    theta = rng.uniform(*theta_range, n)
    fi = rng.uniform(0, 2 * np.pi, n)

    direction = np.stack((np.sin(theta) * np.cos(fi),
                          np.cos(theta),
                          np.sin(theta) * np.sin(fi)), axis=1)

    # путь от источника до плоскости старта вдоль луча
    path = (start_y - source_y) / np.cos(theta)
    r = np.array([0, source_y, 0]) + direction * path[:, None]
    v = direction * speed[:, None]
    return r, v
