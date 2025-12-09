"""Операции над векторами (x, y, z) в виде кортежей.

Кортежи не выделяют память, поэтому внутри numba они намного
быстрее маленьких массивов numpy.
"""

from numba import njit


@njit(cache=True)
def add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@njit(cache=True)
def scale(a, k):
    return a[0] * k, a[1] * k, a[2] * k


@njit(cache=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
