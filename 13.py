import math
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return (math.log(5 * x ** 4 - 3 * x ** 2) / (2 * x ** 2 + math.sin(x / 4))) + (
            (3 * x ** 2 - math.cos(3 * x)) / math.log(x ** 3 + 3))


def df(x, h=0):
    # u = ln(5x^4 - 3x^2) / (2x^2 + sin(x/4))
    p = math.log(5 * x ** 4 - 3 * x ** 2)
    q = 2 * x ** 2 + math.sin(x / 4)
    p_prime = (20 * x ** 3 - 6 * x) / (5 * x ** 4 - 3 * x ** 2)
    q_prime = 4 * x + 0.25 * math.cos(x / 4)
    u_prime = (p_prime * q - p * q_prime) / (q ** 2)

    # v = (3x^2 - cos(3x)) / ln(x^3 + 3)
    r = 3 * x ** 2 - math.cos(3 * x)
    s = math.log(x ** 3 + 3)
    r_prime = 6 * x + 3 * math.sin(3 * x)
    s_prime = 3 * x ** 2 / (x ** 3 + 3)
    v_prime = (r_prime * s - r * s_prime) / (s ** 2)

    return u_prime + v_prime


def d2f(x, h=0):
    # Для u''(x)
    p = math.log(5 * x ** 4 - 3 * x ** 2)
    q = 2 * x ** 2 + math.sin(x / 4)
    p_prime = (20 * x ** 3 - 6 * x) / (5 * x ** 4 - 3 * x ** 2)
    q_prime = 4 * x + 0.25 * math.cos(x / 4)

    # p'' и q''
    p_double = (60 * x ** 2 - 6) * (5 * x ** 4 - 3 * x ** 2) - (20 * x ** 3 - 6 * x) * (20 * x ** 3 - 6 * x)
    p_double /= (5 * x ** 4 - 3 * x ** 2) ** 2
    q_double = 4 - 0.0625 * math.sin(x / 4)  # производная cos(x/4)/4 = -sin(x/4)/16

    # u''(x) = (p'q - pq')'/q^2 - 2q'q (p'q - pq') / q^4
    numerator = (p_double * q + p_prime * q_prime - (p_prime * q_prime + p * q_double)) * q ** 2 - 2 * q * q_prime * (
            p_prime * q - p * q_prime)
    u_double = numerator / (q ** 4)

    # Для v''(x)
    r = 3 * x ** 2 - math.cos(3 * x)
    s = math.log(x ** 3 + 3)
    r_prime = 6 * x + 3 * math.sin(3 * x)
    s_prime = 3 * x ** 2 / (x ** 3 + 3)

    # r'' и s''
    r_double = 6 + 9 * math.cos(3 * x)
    s_double = (6 * x * (x ** 3 + 3) - 9 * x ** 4) / (x ** 3 + 3) ** 2  # по формуле d^2 ln(x^3+3)/dx^2

    # v''(x) = (r's - rs')'/s^2 - 2s's (r's - rs') / s^4
    numerator_v = ((r_double * s + r_prime * s_prime) - (
            r_prime * s_prime + r * s_double)) * s ** 2 - 2 * s * s_prime * (r_prime * s - r * s_prime)
    v_double = numerator_v / (s ** 4)

    return u_double + v_double


def d3f(x, h):
    if round(x - h, 4) < round(x_start, 4):
        return None
    return (d2f(x) - d2f(x - h)) / h


def d4f(x, h):
    if round(x - h, 4) < round(x_start, 4):
        return None
    return (d3f(x, h) - d3f(x - h, h)) / h


def d5f(x, h):
    if round(x - h, 4) < round(x_start, 4):
        return None
    return (d4f(x, h) - d4f(x - h, h)) / h


def d6f(x, h):
    if round(x - h, 4) < round(x_start, 4):
        return None
    return (d5f(x, h) - d5f(x - h, h)) / h


def find_max(func, a, b, h):
    n = int((b - a) / h)
    x_val = np.linspace(a, b, n)
    mx = 0
    for x in x_val:
        mx = max(x, func(x))
    return mx


x_start = 1.0
x_end = 4.0


def two_point_1(x, h):
    if round(x - h, 4) < round(x_start, 4):
        return None
    return (f(x) - f(x - h)) / h


def two_point_1_error(x, h):
    return h / 2 * abs(d2f(x, h))


def two_point_2(x, h):
    if round(x + h, 4) > round(x_end):
        return None
    return (f(x + h) - f(x)) / h


def two_point_3(x, h):
    if round(x + h, 4) > round(x_end, 4) or round(x - h, 4) < round(x_start, 4):
        return None
    return (f(x + h) - f(x - h)) / (2 * h)


def two_point_3_error(x, h):
    return h * h / 6 * abs(d2f(x, h))


def three_point_4(x, h):
    if round(x + 2 * h, 4) > round(x_end, 4):
        return None
    return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)


def three_point_4_error(x, h):
    return h * h / 3 * abs(d3f(x, h))


def four_point_9(x, h):
    if round(x + 2 * h, 4) > round(x_end, 4) or round(x - h, 4) < round(x_start, 4):
        return None
    return (-2 * f(x - h) - 3 * f(x) + 6 * f(x + h) - f(x + 2 * h)) / (6 * h)


def four_point_9_error(x, h):
    return h ** 3 / 12 * abs(d3f(x, h))


def five_point_19(x, h):
    if round(x + 4 * h, 4) > round(x_end, 4):
        return None
    return (-25 * f(x) + 48 * f(x + h) - 36 * f(x + 2 * h) + 16 * f(x + 3 * h) - 3 * f(x + 4 * h)) / (12 * h)


def five_point_19_error(x, h):
    return h ** 4 / 5 * abs(d5f(x, h))


def three_point_d2f_7(x, h):
    if round(x - 2 * h, 4) < round(x_start, 4):
        return None
    return (f(x - 2 * h) - 2 * f(x - h) + f(x)) / (h * h)


def three_point_d2f_7_error(x, h):
    return h * abs(d3f(x, h))


def three_point_d2f_8(x, h):
    if round(x - h, 4) < round(x_start, 4) or round(x + h, 4) > round(x_end, 4):
        return None
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def three_point_d2f_8_error(x, h):
    return h * h / 12 * abs(d4f(x, h))


def four_point_d2f_16(x, h):
    if round(x + 3 * h, 4) > round(x_end, 4):
        return None
    return (2 * f(x) - 5 * f(x + h) + 4 * f(x + 2 * h) - f(x + 3 * h)) / (h * h)


def four_point_d2f_16_error(x, h):
    return 11 / 12 * h * h * abs(d4f(x, h))


def five_point_d2f_23(x, h):
    if round(x + 2 * h, 4) > round(x_end, 4) or round(x - 2 * h, 4) < round(x_start, 4):
        return None
    return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * h * h)


def five_point_d2f_23_error(x, h):
    return h ** 4 / 90 * abs(d6f(x, h))


def derivative(f_scheme, x, h):
    return f_scheme(x, h)


def print_values(derivative_schemes, top_header, x_values, order, h):
    names = [i[1] for i in derivative_schemes]
    col_width = 20
    header = "{:<10}".format("x")
    print(f"{top_header}. Шаг сетки h = {h}")

    for name in names:
        header += "{:<{width}}".format(name, width=col_width)
    print(header)

    num_values = {name: [] for name in names}
    for x in x_values:
        res_str = "{:<10.2f}".format(x)  # выводим x с двумя знаками после запятой

        for func, name, error in derivative_schemes:
            try:
                res = derivative(func, x, h)
                res_str += "{:<{width}.8f}".format(res, width=col_width)
                num_values[name].append((x, res))
            except Exception:
                res_str += "{:<{width}}".format("-", width=col_width)

        print(res_str)

    print('\n')

    for name in names:
        x = [i[0] for i in num_values[name]]
        y = [i[1] for i in num_values[name]]
        plt.plot(x, y, label=name)

    if order == 1:
        df_values = [df(x) for x in x_values]
    elif order == 2:
        df_values = [d2f(x) for x in x_values]
    else:
        raise Exception("Неверный порядок")

    plt.plot(x_values, df_values, label=f"Analytical", marker='o')
    plt.title(f"{top_header}. Шаг сетки h = {h}")
    plt.xlabel("x")
    if order == 1:
        plt.ylabel("f'(x)")
    elif order == 2:
        plt.ylabel("f''(x)")
    plt.grid(True)
    plt.legend()
    plt.show()
    draw_errors(derivative_schemes, top_header, x_values, order, h)


def draw_errors(derivative_schemes, top_header, x_values, order, h):
    x_values = x_values[5:]
    names = [i[1] for i in derivative_schemes]
    num_values = {name: [] for name in names}
    for func, name, error in derivative_schemes:
        y_values = []
        for x in x_values:
            try:
                res = error(x, h)
                num_values[name].append((x, res))
            except Exception:
                pass
    for name in names:
        x = [i[0] for i in num_values[name]]
        y = [i[1] for i in num_values[name]]
        plt.plot(x, y, label=f"{name} error")
    plt.title(f"Ошибка {top_header}. Шаг сетки h = {h}")
    plt.xlabel("x")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    h = 0.15

    num_points = int((x_end - x_start) / h) + 1
    x_values = np.linspace(x_start, x_end, num_points)

    derivative_schemes = [
        (two_point_1, "One-point var 1", two_point_1_error),
        (two_point_2, "One-point var 2", two_point_1_error),
        (two_point_3, "One-point var 3", two_point_3_error),
        (three_point_4, "Three-point var 4", three_point_4_error),
        (four_point_9, "Four-point var 9", four_point_9_error),
        (five_point_19, "Five-point var 19", five_point_19_error)
    ]

    print_values(derivative_schemes, "Первая производная", x_values, 1, h)
    print_values(derivative_schemes, "Первая производная", x_values, 1, h / 2.0)

    derivative_schemes2 = [
        (three_point_d2f_7, "Three-point var 7", three_point_d2f_7_error),
        (three_point_d2f_8, "Three-point var 8", three_point_d2f_8_error),
        (four_point_d2f_16, "Four-point var 16", four_point_d2f_16_error),
        (five_point_d2f_23, "Five-point var 23", five_point_d2f_23_error)
    ]
    print_values(derivative_schemes2, "Вторая производная", x_values, 2, h)
    print_values(derivative_schemes2, "Вторая производная", x_values, 2, h / 2.0)


if __name__ == "__main__":
    main()
