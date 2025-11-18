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
    z_plot = []
    for i in x_plot:
        z_plot.append(d2f(i))
    plt.plot(x_plot, y_plot, label='f(x)')
    # plt.plot(x_plot, z_plot, label='d2f(x)')
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
        raise Exception("Не выполнены условия для метода")

    it = 0
    while b - a > EPSILON:
        x = (b + a) / 2
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
        it += 1

    return (a + b) / 2, it


def simple_iteration(a, b, l):
    max_dphi = max(abs(dphi(xi, l)) for xi in [a, b])
    if f(a) * f(b) >= 0 and max_dphi >= 1:
        return None, 0

    it = 0

    x = 0
    d_phi_a = abs(dphi(a, l))
    d_phi_b = abs(dphi(b, l))
    q = max(d_phi_a, d_phi_b)
    if (int(q) == 1):
        print("Сходимость не гарантирована (q = 1)!")
        if d_phi_a < 1:
            x = a
        elif d_phi_b < 1:
            x = b
        else:
            x = a
    elif q > 1:
        raise Exception(f"|φ'(x)| ≤ q < 1 не выполнено (q = {q:.3f})!")

    if d_phi_a < 1:
        x = a
    elif d_phi_b < 1:
        x = b

    while True:
        it += 1
        x_new = phi(x, l)
        if abs(x - x_new) < EPSILON:
            return x_new, it
        x = x_new



def newton(a, b):
    if f(a) * f(b) >= 0:
        raise Exception("Не выполнены условия для метода")

    if df(a) == 0 or df(b) == 0:
        raise Exception("Не выполнены условия для метода")

    if (f(a) * d2f(a) < df(a) ** 2) and \
            (f(b) * d2f(a) < df(b) ** 2):
        if abs(f(a)) < abs(f(b)):
            x = a
        else:
            x = b
    elif f(a) * d2f(a) < df(a) ** 2:
        x = a
    elif f(b) * d2f(b) < df(b) ** 2:
        x = b
    else:
        raise Exception("Не выполнены условия для метода")

    it = 0
    while True:
        x_new = x - f(x) / df(x)
        it += 1
        if abs(x_new - x) < EPSILON:
            return x_new, it
        x = x_new


def secant(a, b):
    if f(a) * f(b) >= 0:
        raise Exception("Не выполнены условия для метода")

    if df(a) == 0 or df(b) == 0:
        raise Exception("Не выполнены условия для метода")

    if (f(a) * d2f(a) < df(a) ** 2) and \
            (f(b) * d2f(a) < df(b) ** 2):
        if abs(f(a)) < abs(f(b)):
            x = a
        else:
            x = b

    if f(a) * d2f(a) < df(a) ** 2:
        x = a
    elif f(b) * d2f(b) < df(b) ** 2:
        x = b
    else:
        raise Exception("Не выполнены условия для метода")

    it = 0
    x_prev = x

    x = x_prev - f(x_prev) / df(x_prev)

    while abs(x - x_prev) > EPSILON:
        x_new = x - f(x) * (x - x_prev) / (f(x) - f(x_prev))
        x, x_prev = x_new, x
        it += 1

    return x, it


def hord(a, b):
    if f(a) * f(b) >= 0:
        raise Exception("Не выполнены условия для метода")

    if df(a) == 0 or df(b) == 0:
        raise Exception("Не выполнены условия для метода")

    if (f(a) * d2f(a) < df(a) ** 2) and \
            (f(b) * d2f(a) < df(b) ** 2):
        if abs(f(a)) < abs(f(b)):
            x = a
            z = b
        else:
            x = b
            z = a

    if f(a) * d2f(a) < df(a) ** 2:
        x = a
        z = b
    elif f(b) * d2f(b) < df(b) ** 2:
        x = b
        z = a
    else:
        raise Exception("Не выполнены условия для метода")

    it = 0
    while True:
        it += 1
        x_new = x - f(x) * (z - x) / (f(z) - f(x))
        if abs(x_new - x) < EPSILON:
            return x_new, it
        x = x_new


def calc(metod, a, b, l=0.0):
    if metod == simple_iteration:
        root, iter = metod(a, b, l)
    elif metod == dichotomy:
        root, iter = metod(a, b)
    else:
        root, iter = metod(a, b)

    if root is None:
        print("Условия не выполнены")
    else:
        print(f"\tКорень {root:.7g} был найден за {iter} итераци")
        print(f"\t Проверка: f(root) = {round(f(root), 2)}")


def check_conditions(f, a, b, l):
    x0 = 0
    # 1) f(a)*f(b) < 0
    if f(a) * f(b) >= 0:
        raise Exception(f"Условие f(a)*f(b)<0 не выполнено на [{a}, {b}]!")

    if df(a) == 0 or df(b) == 0:
        raise Exception(f"Условия не выполнены")

    # 2) |φ'(x)| ≤ q < 1
    d_phi_a = abs(dphi(a, l))
    d_phi_b = abs(dphi(b, l))

    if d_phi_a < 1 and d_phi_b < 1:
        x0 = min(a, b, key=lambda x: abs(x))
    else:
        return False, 0

    return True, x0


def main():
    a_neg, b_neg = -3.79, -3.5
    a_pos, b_pos = 1.2, 1.3
    c, d = -3.0, 0

    draw()
    print("Деление пополам:")

    ints = [{'a': a_neg, 'b': b_neg},
            {'a': a_pos, 'b': b_pos},
            {'a': c, 'b': d}]
    for i in ints:
        calc(dichotomy, i['a'], i['b'])

    print("Простая итерация:")
    calc(simple_iteration, a_neg, b_neg, 0.1)
    calc(simple_iteration, a_pos, b_pos, 0.1)
    calc(simple_iteration, c, d, -0.1)

    print("Метод Ньютона:")
    for interval in ints:
        calc(newton, interval['a'], interval['b'])

    print("Метод секущих:")
    for interval in ints:
        calc(secant, interval['a'], interval['b'])

    print("Метод хорд:")
    for interval in ints:
        calc(hord, interval['a'], interval['b'])


if __name__ == "__main__":
    main()
