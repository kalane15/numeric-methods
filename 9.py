import dataclasses

import matplotlib.pyplot as plt
import numpy as np

x_table = [3.05, 3.43, 3.81, 4.19, 4.57, 4.95, 5.33, 5.71, 6.09]
y_table = [1.8571, 2.1247, 3.6456, 2.6842, 2.3539, 0.3431, 1.6577, 2.8982, 1.4326]


def plot_multiple_graphs(data):
    num_graphs = len(data)

    for i, (label, x, y, x_star, y_star, x_p, y_p) in enumerate(data):
        plt.subplot(num_graphs, 1, i + 1)
        plt.plot(x, y, label=label)
        if x_star is not None and y_star is not None:
            plt.scatter(x_star, y_star, color='red', zorder=5, label="x*")
        plt.scatter(x_p, y_p, color='blue', zorder=5, label="input")

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"График {label}")

    plt.tight_layout()
    plt.show()


def l_i(x, x_vals, i):
    """
    Lagrange basis polynomial l_i(x):
    l_i(x) = ∏_{j=0, j≠i}^n (x - x_j) / (x_i - x_j)
    """
    n = len(x_vals)
    prod = 1.0
    for j in range(n):
        if j != i:
            prod *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
    return prod


def lagrange(x_vals, y_vals, x):
    """
    Interpolation polynomial L_n(x):
    L_n(x) = ∑_{i=0}^n y_i * l_i(x)
    """
    n = len(x_vals)
    result = 0.0
    for i in range(n):
        result += y_vals[i] * l_i(x, x_vals, i)
    return result


def error(x, x_vals):
    prod = 1.0
    for xj in x_vals:
        prod *= (x - xj)
    return prod


def build_lagrange(x_t, y_t, x_star, start, stop):
    x_quad = x_t[start:stop]
    y_quad = y_t[start:stop]

    x_check = x_quad[1]
    y_check = y_quad[1]
    p2_star = lagrange(x_quad, y_quad, x_star)
    p2_check = lagrange(x_quad, y_quad, x_check)

    x = np.linspace(x_t[0], x_t[-1], 100)
    y = []
    for i in x:
        y.append(lagrange(x_quad, y_quad, i))

    omega2 = abs(error(x_star, x_quad))
    print(f"Value of L_{stop-start - 1}(x*) = {p2_star}")
    print(f"Check: L_{stop-start - 1}({x_check}) = {p2_check} (expected: {y_check})")
    print(f"|ω(x*)| = {omega2}")
    print()

    return x, y, omega2, p2_star, x_quad, y_quad


def main():
    x_star = 4.016

    omegas_to_value = dict()

    data = []
    x, y, omega, p_star, points_x, points_y = build_lagrange(x_table, y_table, x_star, 1, 4)
    omegas_to_value[omega] = p_star
    data.append(("L2", x, y, x_star, p_star, points_x, points_y))

    x, y, omega, p_star, points_x, points_y = build_lagrange(x_table, y_table, x_star, 2, 5)
    omegas_to_value[omega] = p_star
    data.append(("L2", x, y, x_star, p_star, points_x, points_y))
    plot_multiple_graphs(data)

    data = []

    x, y, omega, p_star, points_x, points_y = build_lagrange(x_table, y_table, x_star, 2, 6)
    omegas_to_value[omega] = p_star
    data.append(("L3", x, y, x_star, p_star, points_x, points_y))

    x, y, omega, p_star, points_x, points_y = build_lagrange(x_table, y_table, x_star, 1, 5)
    omegas_to_value[omega] = p_star
    data.append(("L3", x, y, x_star, p_star, points_x, points_y))

    x, y, omega, p_star, points_x, points_y = build_lagrange(x_table, y_table, x_star, 0, 4)
    omegas_to_value[omega] = p_star
    data.append(("L3", x, y, x_star, p_star, points_x, points_y))

    plot_multiple_graphs(data)

    min_omega = min(omegas_to_value.keys())

    print(f"Value in x_star with minimum error is {omegas_to_value[min_omega]}")


if __name__ == "__main__":
    main()
