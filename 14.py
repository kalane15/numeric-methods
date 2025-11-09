import math

import numpy as np


# Функция под интегралом
def f(x):
    return math.sinh(math.sin(3 * x)) / ((x + 1) ** 2)


def df(x):
    return (3 * math.cosh(math.sin(3 * x)) * math.cos(3 * x)) / (x + 1) ** 2 - (2 * math.sinh(math.sin(3 * x))) / (
            x + 1) ** 3


def d2f(x):
    return (-9 * math.sinh(math.sin(3 * x)) * math.cos(3 * x) ** 2
            - 3 * math.cosh(math.sin(3 * x)) * math.sin(3 * x)) / (x + 1) ** 2 \
        - (12 * math.cosh(math.sin(3 * x)) * math.cos(3 * x)) / (x + 1) ** 3 \
        + (6 * math.sinh(math.sin(3 * x))) / (x + 1) ** 4


def find_max(func, a, b, h):
    n = int((b - a) / h)
    x_val = np.linspace(a, b, n)
    mx = 0
    for x in x_val:
        mx = max(x, func(x))
    return mx


def d4f(x):
    return (81 * math.sinh(math.sin(3 * x)) * math.cos(3 * x) ** 4
            - 162 * math.cosh(math.sin(3 * x)) * math.sin(3 * x) * math.cos(3 * x) ** 2
            - 27 * math.sinh(math.sin(3 * x)) * (math.sin(3 * x) ** 2 - 2 * math.cos(3 * x) ** 2)
            + 9 * math.cosh(math.sin(3 * x)) * math.sin(3 * x)) / (x + 1) ** 2 \
        + (-108 * math.cosh(math.sin(3 * x)) * math.cos(3 * x) ** 3
           + 108 * math.sinh(math.sin(3 * x)) * math.sin(3 * x) * math.cos(3 * x)
           + 36 * math.cosh(math.sin(3 * x)) * math.cos(3 * x)) / (x + 1) ** 3 \
        - (108 * math.sinh(math.sin(3 * x)) * math.cos(3 * x) ** 2
           + 36 * math.cosh(math.sin(3 * x)) * math.sin(3 * x)) / (x + 1) ** 4 \
        - (48 * math.cosh(math.sin(3 * x)) * math.cos(3 * x)) / (x + 1) ** 5 \
        + (120 * math.sinh(math.sin(3 * x))) / (x + 1) ** 6


# Метод средних прямоугольников
def midpoint_rule(a, b, h):
    n = int((b - a) / h)
    total = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        total += f(x_mid)
    return total * h


# Метод трапеций
def trapezoidal_rule(a, b, h):
    n = int((b - a) / h)
    total = (f(a) + f(b)) / 2
    for i in range(1, n):
        x_i = a + i * h
        total += f(x_i)
    return total * h


# Метод Симпсона (число шагов должно быть чётным)
def simpson_rule(a, b, h):
    n = int((b - a) / h)
    if n % 2 != 0:
        raise Exception("Неккоректный шаг")

    total = f(a) + f(b)
    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            total += 2 * f(x_i)
        else:
            total += 4 * f(x_i)
    return total * h / 3


def euler_rule(a, b, h):
    n = int((b - a) / h)

    total = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x_i = a + i * h
        total += f(x_i)
    total = total * h
    return total + 1.0 / 12.0 * h * h * (df(a) - df(b))


def runge_romberg(I_h, I_h2, p):
    return I_h2 + (I_h2 - I_h) / (2 ** p - 1)


def calculate_mistake(method, a, b, h):
    if method == midpoint_rule:
        mx = find_max(lambda x: abs(d2f(x)), a, b, h)
        return mx * (b - a) * h * h / 24
    elif method == trapezoidal_rule:
        mx = find_max(lambda x: abs(d2f(x)), a, b, h)
        return mx * (b - a) * h * h / 12
    elif method == simpson_rule:
        mx = find_max(lambda x: abs(d4f(x)), a, b, h)
        return mx * (b - a) * h ** 4 / 180
    elif method == euler_rule:
        mx = find_max(lambda x: abs(d4f(x)), a, b, h)
        return mx * h ** 4 / 720
    else:
        raise Exception("Неизвестный метод")


a, b = 0, 1
h = 0.1

methods = [
    ("Средние прямоугольники", midpoint_rule, 2),
    ("Трапеции", trapezoidal_rule, 2),
    ("Симпсона", simpson_rule, 4),
    ("Эйлера", euler_rule, 4)
]

print(f"{'Метод':<25} {'I(h)':>14} {'I(h/2)':>14} {'Уточнённое':>16} {'Погр. при h':>14} {'Погр. при h/2':>14}")
print("=" * 105)

for name, method, p in methods:
    I_h = method(a, b, h / 2)
    I_h2 = method(a, b, h)
    I_refined = runge_romberg(I_h2, I_h, p)
    m1 = calculate_mistake(method, a, b, h)
    m2 = calculate_mistake(method, a, b, h / 2)

    print(f"{name:<25}"
          f"{I_h:14.8f} "
          f"{I_h2:14.8f} "
          f"{I_refined:16.8f} "
          f"{m1:16.8e} "
          f"{m2:16.8e}")
