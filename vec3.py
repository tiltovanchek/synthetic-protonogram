"""Операции над векторами (x, y, z) в виде кортежей.

Кортежи не выделяют память, поэтому внутри numba они намного
быстрее маленьких массивов numpy.
"""

from numba import njit


@njit
def add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@njit
def scale(a, k):
    return a[0] * k, a[1] * k, a[2] * k


@njit
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
