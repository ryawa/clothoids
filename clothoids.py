import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton


# Approximation from http://www.dgp.toronto.edu/~mccrae/projects/clothoid/sbim2008mccrae.pdf
def C(t):
    return 1 / 2 - R(t) * np.sin(1 / 2 * np.pi * (A(t) - t**2))


def S(t):
    return 1 / 2 - R(t) * np.cos(1 / 2 * np.pi * (A(t) - t**2))


def R(t):
    return (0.506 * t + 1) / (1.79 * t**2 + 2.054 * t + np.sqrt(2))


def A(t):
    return 1 / (0.803 * t**3 + 1.886 * t**2 + 2.524 * t + 2)


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def theta_to_t(theta):
    return np.sqrt(2 * theta / np.pi)


def t_to_theta(t):
    return t * t * np.pi / 2


def make_points_feasible(points, tau):
    alpha = angle(points[1] - points[0], points[2] - points[1])
    t = theta_to_t(alpha)
    limit = C(t) / S(t)
    a = np.linalg.norm(points[1] - points[0])
    b = np.linalg.norm(points[2] - points[1])

    if a > b:
        g = a
        h = b
    else:
        points[[0, 2]] = points[[2, 0]]
        g = b
        h = a

    T0 = (points[1] - points[0]) / g

    new_point = points[0]
    if (g / h + np.cos(alpha)) / np.sin(alpha) > limit:
        g_lim = h * (limit * np.sin(alpha) - np.cos(alpha))
        T_lerp = (1 - tau) * h + tau * g_lim
        new_point = points[1] - T_lerp * T0
    return np.insert(points, 1, new_point, axis=0)


def f(theta, alpha, k):
    t0 = theta_to_t(theta)
    t1 = theta_to_t(alpha - theta)
    return np.sqrt(theta) * (
        C(t0) * np.sin(alpha) - S(t0) * (k + np.cos(alpha))
    ) + np.sqrt(alpha - theta) * (
        S(t1) * (1 + k * np.cos(alpha)) - k * C(t1) * np.sin(alpha)
    )


def fprime(theta, alpha, k):
    t0 = theta_to_t(theta)
    t1 = theta_to_t(alpha - theta)
    return S(t0) * np.sin(alpha) / (2 * np.sqrt(theta)) * (
        C(t0) / S(t0) - (k + np.cos(alpha)) / np.sin(alpha)
    ) + (k * S(t1) * np.sin(alpha)) / (2 * np.sqrt(alpha - theta)) * (
        C(t1) / S(t1) - (1 + k * np.cos(alpha)) / (k * np.sin(alpha))
    )


def asymmetric_clothoid(points, tau):
    omega = angle(points[1] - points[0], points[1] - points[2])
    if abs(omega - np.pi) < 1e-6:
        raise ValueError()
        pass

    points = make_points_feasible(points, tau)
    alpha = angle(points[2] - points[1], points[3] - points[2])
    a = np.linalg.norm(points[2] - points[1])
    b = np.linalg.norm(points[3] - points[2])
    g = max(a, b)
    h = min(a, b)
    theta0 = newton(f, alpha / 2, fprime=fprime, args=(alpha, g / h))
    theta1 = alpha - theta0
    t0 = theta_to_t(theta0)
    t1 = theta_to_t(theta1)

    num = g + h * np.cos(alpha)
    scale = np.sqrt(theta1 / theta0)
    denom = C(t0) + scale * C(t1) * np.cos(alpha) + scale * S(t1) * np.sin(alpha)
    a0 = num / denom
    a1 = a0 * scale

    P0 = points[1]
    T0 = (points[2] - points[1]) / g
    P1 = points[3]
    T1 = (points[2] - points[3]) / h

    N0 = np.array([-T0[1], T0[0]])
    N1 = np.array([-T1[1], T1[0]])
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    crossT1T0 = cross(T1, T0)
    if np.sign(cross(T0, N0)) != np.sign(crossT1T0):
        N0 *= -1
    if np.sign(cross(N1, T1)) != np.sign(crossT1T0):
        N1 *= -1

    return (points[0], points[1]), (P0, T0, N0, a0, t0), (P1, T1, N1, a1, t1)


def sample_clothoid(P, T, N, a, t, step):
    points = []
    for x in np.arange(0, t + step, step):
        points.append(P + a * C(x) * T + a * S(x) * N)
    return points


control = [[0, 0], [3, 0], [0, 5]]
plt.plot(*zip(*control), "r--")
L, (P0, T0, N0, a0, t0), (P1, T1, N1, a1, t1) = asymmetric_clothoid(
    np.array(control, dtype=float), 0.5
)
N = 1000
print(
    timeit.timeit(
        lambda: asymmetric_clothoid(np.array(control, dtype=float), 0.5), number=N
    )
    / N
)
A1 = sample_clothoid(P0, T0, N0, a0, t0, 0.01)
A2 = sample_clothoid(P1, T1, N1, a1, t1, 0.01)
plt.plot(*zip(*L))
plt.plot(*zip(*A1))
plt.plot(*zip(*A2))
plt.axis("equal")
plt.show()
