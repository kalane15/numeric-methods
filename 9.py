import dataclasses

import matplotlib.pyplot as plt
import numpy as np

x_table = [3.05, 3.43, 3.81, 4.19, 4.57, 4.95, 5.33, 5.71, 6.09]
y_table = [1.8571, 2.1247, 3.6456, 2.6842, 2.3539, 0.3431, 1.6577, 2.8982, 1.4326]

import matplotlib.pyplot as plt


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


def omega(x, x_vals):
    """
    Polynomial ω_{n+1}(x) = ∏_{j=0}^n (x - x_j).
    """
    prod = 1.0
    for xj in x_vals:
        prod *= (x - xj)
    return prod


def build_lagrange_2(x_t, y_t, x_star, start, stop):
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

    omega2 = abs(omega(x_star, x_quad))
    print(f"Value of L_2(x*) = {p2_star}")
    print(f"Check: L_2({x_check}) = {p2_check} (expected: {y_check})")
    print(f"|ω_3(x*)| = {omega2}")

    return x, y, omega2, p2_star, x_quad, y_quad


def main():
    # Points for the 2nd degree polynomial (n=2, indices i=1,2,3 in the table)
    x_quad = [x_table[1], x_table[2], x_table[3]]
    y_quad = [y_table[1], y_table[2], y_table[3]]

    x_quad2 = [x_table[2], x_table[3], x_table[4]]
    y_quad2 = [y_table[2], y_table[3], y_table[4]]

    # Points for the 3rd degree polynomial (n=3, indices i=1,2,3,4)
    x_cub = [x_table[1], x_table[2], x_table[3], x_table[4]]
    y_cub = [y_table[1], y_table[2], y_table[3], y_table[4]]

    x_star = 4.016
    x_check = x_table[2]
    y_check = y_table[2]
    data = []
    x, y, omega, p_star, points_x, points_y = build_lagrange_2(x_table, y_table, x_star, 1, 4)
    data.append(("L2", x, y, x_star, p_star, points_x, points_y))
    x, y, omega, p_star, points_x, points_y = build_lagrange_2(x_table, y_table, x_star, 2, 5)
    data.append(("L2", x, y, x_star, p_star, points_x, points_y))
    plot_multiple_graphs(data)

    # p3_star = lagrange(x_cub, y_cub, x_star)

    # p3_check = lagrange(x_cub, y_cub, x_check)

    # omega3 = abs(omega(x_star, x_cub))  # For n=3
    # r3_bound = omega3 / 24  # 4! = 24
    # draw(x_quad, y_quad, x_cub, y_cub, x_star, p2_star, p3_star, omega2, omega3, r2_bound, r3_bound)
    #
    # p2_star2 = lagrange(x_quad2, y_quad2, x_star)
    # omega22 = abs(omega(x_star, x_quad2))
    # draw(x_quad2, y_quad2, x_cub, y_cub, x_star, p2_star2, p3_star, omega22, omega3, r2_bound, r3_bound)
    #
    #
    #
    # print(f"Value of L_3(x*) = {p3_star}")
    # print(f"Check: L_3({x_check}) = {p3_check} (expected: {y_check})")
    # print(f"|ω_4(x*)| = {omega3}")
    # print(f"Bound for |R_3(x*)| ≤ (M / 24) * |ω_4(x*)| = {r3_bound} * M")


if __name__ == "__main__":
    main()
