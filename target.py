"""Геометрия мишени: кривая из отрезков с зарядом и током."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Target:
    mid: np.ndarray  # центры отрезков, (M, 3)
    dl: np.ndarray  # векторы отрезков, (M, 3)
    dq: np.ndarray  # заряд каждого отрезка, (M,)
    current: float  # ток вдоль кривой, А

    @property
    def length(self):
        return np.linalg.norm(self.dl, axis=1).sum()


def circle(radius, n_points):
    """Окружность в плоскости x = 0, обход от оси z к оси y."""
    fi = np.linspace(0, 2 * np.pi, n_points)
    y = radius * np.sin(fi)
    z = radius * np.cos(fi)
    return np.stack((np.zeros(n_points), y, z), axis=1)


def snail():
    """Улитка в плоскости x = 0, оцифрована с изображения в пикселях."""
    y = np.array([174, 185, 192, 192, 188, 181, 168, 152, 137, 126, 120, 117, 118, 125,
                  134, 152, 168, 189, 202, 217, 233, 249, 263, 277, 293, 305, 318, 330,
                  342, 354, 363, 372, 377, 378, 377, 372, 367, 360, 351, 339, 327, 312,
                  296, 284, 271, 252, 232, 210, 193, 190, 186, 183, 180, 178, 176, 174])
    z = np.array([330, 315, 300, 289, 277, 266, 253, 239, 226, 211, 195, 177, 162, 147,
                  137, 122, 113, 105, 103, 103, 104, 107, 112, 115, 122, 129, 139, 149,
                  163, 178, 197, 218, 238, 266, 287, 309, 328, 347, 366, 389, 409, 428,
                  446, 457, 467, 480, 490, 502, 509, 481, 452, 425, 401, 377, 357, 330])
    pixel = 91.2e-6 / 180  # размер пикселя, м
    y = (y - 174) * pixel
    z = (330 - z) * pixel

    # поворот на -55 градусов вокруг оси x
    a = np.radians(-55)
    y, z = y * np.cos(a) - z * np.sin(a), y * np.sin(a) + z * np.cos(a)
    return np.stack((np.zeros(len(y)), y, z), axis=1)


def build_target(points, charge, current):
    """Разбивает кривую на отрезки, заряд делится пропорционально длине."""
    dl = points[1:] - points[:-1]
    mid = 0.5 * (points[1:] + points[:-1])
    lengths = np.linalg.norm(dl, axis=1)
    dq = charge * lengths / lengths.sum()
    return Target(mid, dl, dq, current)
