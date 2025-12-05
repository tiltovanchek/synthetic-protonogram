"""Вывод синтетической протонограммы."""

import matplotlib.pyplot as plt


def plot_screen(x, z, title, limit_mm):
    """Точки попадания протонов на экран, координаты в метрах."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x * 1e3, z * 1e3, s=0.7, color="b")
    ax.set_title(title)
    ax.set_xlabel("x, мм", fontsize=15)
    ax.set_ylabel("z, мм", fontsize=15)
    ax.set_xlim(-limit_mm, limit_mm)
    ax.set_ylim(-limit_mm, limit_mm)
    ax.set_aspect("equal")
    ax.grid(lw=0.5, ls="--")
    return fig
