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


def simple_iteration(a, b, l):
    it = 0
    ok, x = check_conditions(f, a, b, l)
    while True:
        it += 1
        x_new = phi(x, l)
        if abs(x - x_new) < EPSILON:
            return x_new, it
        x = x_new


def newton(a, b, x0):
    if f(a) * f(b) >= 0:
        return None, 0
    x_n = np.linspace(a, b, 100)
    for xi in x_n:
        if abs(f(xi) * d2f(xi)) >= df(xi) ** 2 or df(xi) == 0:
            print("sad")
            return None, 0

    it = 0
    x = x0
    while True:
        x_new = x - f(x) / df(x)
        it += 1
        if abs(x_new - x) < EPSILON:
            return x_new, it
        x = x_new


def secant(a, b, x0):
    if f(a) * f(b) >= 0:
        return None, 0
    x_n = np.linspace(a, b, 100)
    for xi in x_n:
        if abs(f(xi) * d2f(xi)) >= df(xi) ** 2 or df(xi) == 0:
            return None, 0

    it = 0
    x_prev = x0

    x = x_prev - f(x_prev) / df(x_prev)

    while abs(x - x_prev) > EPSILON:
        x_new = x - f(x) * (x - x_prev) / (f(x) - f(x_prev))
        x, x_prev = x_new, x
        it += 1

    return x, it


def find_root_intervals(x_start, x_end, dx=0.5) -> list:
    intervals = []
    x_values = np.arange(x_start, x_end + dx, dx)

    for i in range(len(x_values) - 1):
        a = x_values[i]
        b = x_values[i + 1]
        fa = f(a)
        fb = f(b)

        if fa * fb < 0:  # проверка изменения знака
            fpp_a = d2f(a)
            fpp_b = d2f(b)

            cond_a = abs(fa * fpp_a) < (d2f(a)) ** 2
            cond_b = abs(fb * fpp_b) < (d2f(b)) ** 2
            if not (cond_a or cond_b):
                print("Достаточное условие сходимости не выполнено, сходимость не гарантируется!")

            if fa * fpp_a > 0 and fb * fpp_b > 0:
                if abs(fa) <= abs(fb):
                    x0 = a
                else:
                    x0 = b
            elif fa * fpp_a > 0:
                x0 = a
            elif fb * fpp_b > 0:
                x0 = b

            intervals.append({'a': a, 'b': b, 'x0': x0})

    return intervals


def hord(a, b, x0):
    if f(a) * f(b) >= 0:
        return None, 0
    x_n = np.linspace(a, b, 100)
    for xi in x_n:
        if abs(f(xi) * d2f(xi)) >= df(xi) ** 2:
            return None, 0

    if x0 == a:
        z = a
        x = b
    elif x0 == b:
        z = b
        x = a
    else:
        if abs(f(a)) <= abs(f(b)):
            z = x0
            x = a
        else:
            z = x0
            x = b

    it = 0
    while True:
        it += 1
        x_new = x - f(x) * (z - x) / (f(z) - f(x))
        if abs(x_new - x) < EPSILON:
            return x_new, it
        x = x_new


def calc(metod, a, b, x0=0.0, l=0.0):
    if metod == simple_iteration:
        root, iter = metod(a, b, l)
    elif metod == dichotomy:
        root, iter = metod(a, b)
    else:
        root, iter = metod(a, b, x0)

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

    # 2) |φ'(x)| ≤ q < 1
    d_phi_a = abs(dphi(a, l))
    d_phi_b = abs(dphi(b, l))
    q = max(d_phi_a, d_phi_b)
    if (int(q) == 1):
        print("Сходимость не гарантирована (q = 1)!")
        if d_phi_a < 1:
            x0 = a
        elif d_phi_b < 1:
            x0 = b
        else:
            x0 = a
        return True, x0
    elif q > 1:
        raise Exception(f"|φ'(x)| ≤ q < 1 не выполнено (q = {q:.3f})!")

    if d_phi_a < 1:
        x0 = a
    elif d_phi_b < 1:
        x0 = b

    print(f"Условия f(a)*f(b)<0, |φ'(x)|={q:.3f}<1 выполнены.")
    return True, x0


def main():
    a_neg, b_neg = -3.79, -3.5
    a_pos, b_pos = 1.1, 1.3
    c, d = -1.1, -0.5
    draw()
    print("Деление пополам:")
    calc(dichotomy, a_neg, b_neg)
    calc(dichotomy, a_pos, b_pos)
    calc(dichotomy, c, d)

    print("Простая итерация:")
    calc(simple_iteration, a_neg, b_neg, l=0.1)
    calc(simple_iteration, a_pos, b_pos, l=0.1)
    calc(simple_iteration, c, d, l=-0.1)

    ints = find_root_intervals(-6.0, 4.0)

    if len(ints) == 0:
        raise Exception("нет подходящих интервалов")

    print("Метод Ньютона:")

    for interval in ints:
        calc(newton, interval['a'], interval['b'], interval['x0'])

    print("Метод секущих:")
    for interval in ints:
        calc(secant, interval['a'], interval['b'], interval['x0'])

    print("Метод хорд:")
    for interval in ints:
        calc(hord, interval['a'], interval['b'], interval['x0'])


if __name__ == "__main__":
    main()
