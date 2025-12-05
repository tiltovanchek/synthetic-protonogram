"""Синтетическая протонограмма мишени с зарядом и током.

Запуск: python main.py
Параметры задаются в config.py.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from plot import plot_screen
from simulation import project_to_screen, track
from source import generate_protons
from target import build_target, circle, snail


def make_points():
    if cfg.TARGET_SHAPE == "circle":
        return circle(cfg.CIRCLE_RADIUS, cfg.CIRCLE_POINTS)
    if cfg.TARGET_SHAPE == "snail":
        return snail()
    raise ValueError(f"Неизвестная форма мишени: {cfg.TARGET_SHAPE}")


def main():
    start = time.time()
    rng = np.random.default_rng(cfg.SEED)

    target = build_target(make_points(), cfg.TARGET_CHARGE, cfg.TARGET_CURRENT)
    print(f"Длина мишени: {target.length * 1e6:.1f} мкм")

    r, v = generate_protons(cfg.N_PROTONS, rng,
                            energy_mev=(cfg.ENERGY_MIN, cfg.ENERGY_MAX),
                            theta_range=(cfg.THETA_MIN, cfg.THETA_MAX),
                            source_y=cfg.SOURCE_Y, start_y=cfg.START_Y,
                            mass=cfg.M_PROTON, mev=cfg.MEV)
    r, v = track(r, v, target, cfg.DT, cfg.START_Y, cfg.EXIT_Y,
                 cfg.Q_OVER_M, cfg.MAX_STEPS)
    x, z = project_to_screen(r, v, cfg.SCREEN_Y)
    print(f"Время счета: {time.time() - start:.2f} с")

    title = (f"I = {cfg.TARGET_CURRENT / 1e3:g} кА, "
             f"Q = {cfg.TARGET_CHARGE * 1e9:g} нКл, N = {cfg.N_PROTONS}")
    plot_screen(x, z, title, cfg.PLOT_LIMIT_MM)
    plt.show()


if __name__ == "__main__":
    main()
