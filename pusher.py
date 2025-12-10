"""Шаг движения протона под действием силы Лоренца методом Бориса.

Почему Борис: это стандартный метод для заряда в полях E и B.
Поля считаются один раз за шаг, точность второго порядка,
а в магнитном поле скорость только поворачивается, поэтому
энергия не набирается и не теряется из-за ошибок счета.
"""

from numba import njit

from vec3 import add, cross, dot, scale


@njit
def boris_step(r, v, E, B, dt, q_over_m):
    """Сдвигает координату r и скорость v, кортежи (x, y, z), на шаг dt."""
    h = 0.5 * q_over_m * dt

    v_minus = add(v, scale(E, h))  # половина разгона полем E
    t = scale(B, h)  # поворот в поле B
    s = scale(t, 2.0 / (1.0 + dot(t, t)))
    v_prime = add(v_minus, cross(v_minus, t))
    v_plus = add(v_minus, cross(v_prime, s))
    v_new = add(v_plus, scale(E, h))  # вторая половина разгона

    return add(r, scale(v_new, dt)), v_new
