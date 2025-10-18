import numpy as np
import matplotlib.pyplot as plt
from math import log, e, exp

LN3 = log(3, e)
EPSILON = 1e-4


def f(x):
    return 5 - e ** x - x ** 3 - 3 * x ** 2 + 4 * x


def df(x: float) -> float:
    return -e ** x - 3 * x ** 2 - 6 * x + 4


def d2f(x):
    return -e ** x - 6 * x - 6


def phi(x: float, l) -> float:
    return x + l * f(x)


def dphi(x: float, l) -> float:
    return 1 + l * df(x)


def draw() -> None:
    x_plot = np.linspace(-6, 4, 10000)
    y_plot = f(x_plot)

    plt.plot(x_plot, y_plot, label='f(x)')
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции f(x)')
    plt.legend()
    plt.show()


def draw2() -> None:
    x_plot = np.linspace(-10, 10, 10000)
    y_plot = []
    z_plot = []
    for i in x_plot:
        # y_plot.append(abs(f(i) * d2f(i)))
        z_plot.append(df(i))
    # plt.plot(x_plot, y_plot, label='abs(f(i) * d2f(i))')
    plt.plot(x_plot, z_plot, label='df(x)')
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции f(x)')
    plt.legend()
    plt.show()


def dichotomy(a, b) -> tuple[int, int]:
    if f(a) * f(b) >= 0:
        return None, 0

    it = 0
    while b - a > EPSILON:
        x = (b + a) / 2
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
        it += 1

    return (a + b) / 2, it


def simple_iteration(a, b, l) -> tuple[int, int]:
    x_n = np.linspace(a, b, 100)
    max_dphi = max(dphi(xi, l) for xi in x_n)
    if max_dphi >= 1 or f(a) * f(b) >= 0:
        print(max_dphi)
        return None, 0

    it = 0
    x = phi((a + b) / 2, l)
    while True:
        it += 1
        x_new = phi(x, l)
        if abs(x - x_new) < EPSILON:
            return x_new, it
        x = x_new


def newton(a, b) -> tuple[int, int]:
    if f(a) * f(b) >= 0:
        return None, 0
    x_n = np.linspace(a, b, 100)
    for xi in x_n:
        if abs(f(xi) * d2f(xi)) >= df(xi) ** 2:
            print("sad")
            return None, 0

    x = a if f(a) * d2f(a) > 0 else b
    it = 0
    while True:
        x_new = x - f(x) / df(x)
        it += 1
        if abs(x_new - x) < EPSILON:
            return x_new, it
        x = x_new


def secant(a, b) -> tuple[int, int]:
    if f(a) * f(b) >= 0:
        return None, 0
    x_n = np.linspace(a, b, 100)
    for xi in x_n:
        if abs(f(xi) * d2f(xi)) >= df(xi) ** 2:
            return None, 0

    it = 0
    x_prev = a if f(a) * d2f(a) > 0 else b
    x = x_prev - f(x_prev) / df(x_prev)

    while abs(x - x_prev) > EPSILON:
        x_new = x - f(x) * (x - x_prev) / (f(x) - f(x_prev))
        x, x_prev = x_new, x
        it += 1

    return x, it


def hord(a, b) -> tuple[int, int]:
    if f(a) * f(b) >= 0:
        return None, 0
    x_n = np.linspace(a, b, 100)
    for xi in x_n:
        if abs(f(xi) * d2f(xi)) >= df(xi) ** 2:
            return None, 0

    z = a if f(a) * d2f(a) > 0 else b
    x = a if z == b else b

    it = 0
    while True:
        it += 1
        x_new = x - f(x) * (z - x) / (f(z) - f(x))
        if abs(x_new - x) < EPSILON:
            return x_new, it
        x = x_new


def calc(metod, a, b, l=0.0, phi=None, dphi=None):
    if metod == simple_iteration:
        root, iter = metod(a, b, l)
    else:
        root, iter = metod(a, b)
    print(f"\tThe root {root:.7g} was found in {iter} iterations")
    print(f"\tf(root) = {f(root):.7g}")


def main():
    a_neg, b_neg = -3.79, -3.5
    a_pos, b_pos = 1.1, 1.3
    c, d = -1.1, -0.5
    draw()
    print(f"An initial approximation for the roots [{a_neg}, {b_neg}] and [{a_pos}, {b_pos}]\n")

    print("Dichotomy:")
    calc(dichotomy, a_neg, b_neg)
    calc(dichotomy, a_pos, b_pos)
    calc(dichotomy, c, d)

    print("Simple iteration:")
    calc(simple_iteration, a_neg, b_neg, 0.1)
    calc(simple_iteration, a_pos, b_pos, 0.1)
    calc(simple_iteration, c, d, -0.1)

    print("Newton:")
    calc(newton, a_neg, b_neg)
    calc(newton, a_pos, b_pos)
    calc(newton, c, d)

    print("Secant:")
    calc(secant, a_neg, b_neg)
    calc(secant, a_pos, b_pos)
    calc(secant, c, d)

    print("Hord:")
    calc(hord, a_neg, b_neg)
    calc(hord, a_pos, b_pos)
    calc(hord, c, d)


if __name__ == "__main__":
    main()
