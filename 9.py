import matplotlib.pyplot as plt
import numpy as np

x_table = [3.05, 3.43, 3.81, 4.19, 4.57, 4.95, 5.33, 5.71, 6.09]
y_table = [1.8571, 2.1247, 3.6456, 2.6842, 2.3539, 0.3431, 1.6577, 2.8982, 1.4326]


def draw(x_quad, y_quad, x_cub, y_cub, x_star, p2_star, p3_star, omega2, omega3, r2_bound, r3_bound):
    x_plot = np.linspace(x_table[0], x_table[-1], 200)

    y_quad_plot = [lagrange(x_quad, y_quad, x) for x in x_plot]
    y_cub_plot = [lagrange(x_cub, y_cub, x) for x in x_plot]

    abs_omega2_plot = [abs(omega(x, x_quad)) for x in x_plot]
    abs_omega3_plot = [abs(omega(x, x_cub)) for x in x_plot]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(x_plot, y_quad_plot, 'b-', label='L₂(x)', linewidth=2)
    ax1.scatter(x_table, y_table, color='red', s=50, label='Table points', zorder=5)
    ax1.scatter(x_quad, y_quad, color='green', s=100, marker='o', label='Nodes for L₂', zorder=5)
    ax1.scatter(x_star, p2_star, color='purple', s=80, marker='*', label=f'x* = {x_star}, L₂(x*) ≈ {p2_star:.4f}',
                zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Lagrange Interpolation Polynomial of 2nd Degree')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_plot, abs_omega2_plot, 'm--', label='|ω₃(x)|', linewidth=2)
    ax2.scatter(x_star, omega2, color='purple', s=80, marker='*', label=f'|ω₃(x*)| = {omega2:.6f}')
    ax2.axhline(y=r2_bound, color='orange', linestyle=':', label=f'Bound for |R₂| / M = {r2_bound:.6f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|ω₃(x)|')
    ax2.set_title('Polynomial for Error Estimation (n=2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()

    _, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))

    ax3.plot(x_plot, y_cub_plot, 'g-', label='L₃(x)', linewidth=2)
    ax3.scatter(x_table, y_table, color='red', s=50, label='Table points', zorder=5)
    ax3.scatter(x_cub, y_cub, color='orange', s=100, marker='s', label='Nodes for L₃', zorder=5)
    ax3.scatter(x_star, p3_star, color='purple', s=80, marker='*', label=f'x* = {x_star}, L₃(x*) ≈ {p3_star:.4f}',
                zorder=5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Lagrange Interpolation Polynomial of 3rd Degree')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(x_plot, abs_omega3_plot, 'm--', label='|ω₄(x)|', linewidth=2)
    ax4.scatter(x_star, omega3, color='purple', s=80, marker='*', label=f'|ω₄(x*)| = {omega3:.6f}')
    ax4.axhline(y=r3_bound, color='orange', linestyle=':', label=f'Bound for |R₃| / M = {r3_bound:.6f}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('|ω₄(x)|')
    ax4.set_title('Polynomial for Error Estimation (n=3)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

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


def main():
    # Points for the 2nd degree polynomial (n=2, indices i=1,2,3 in the table)
    x_quad = [x_table[1], x_table[2], x_table[3]]
    y_quad = [y_table[1], y_table[2], y_table[3]]

    # Points for the 3rd degree polynomial (n=3, indices i=1,2,3,4)
    x_cub = [x_table[1], x_table[2], x_table[3], x_table[4]]
    y_cub = [y_table[1], y_table[2], y_table[3], y_table[4]]

    x_star = 4.016
    x_check = x_table[2]
    y_check = y_table[2]

    p2_star = lagrange(x_quad, y_quad, x_star)
    p3_star = lagrange(x_cub, y_cub, x_star)
    p2_check = lagrange(x_quad, y_quad, x_check)
    p3_check = lagrange(x_cub, y_cub, x_check)

    omega2 = abs(omega(x_star, x_quad))  # For n=2
    r2_bound = omega2 / 6  # 3! = 6

    omega3 = abs(omega(x_star, x_cub))  # For n=3
    r3_bound = omega3 / 24  # 4! = 24

    print(f"Value of L_2(x*) = {p2_star}")
    print(f"Check: L_2({x_check}) = {p2_check} (expected: {y_check})")
    print(f"|ω_3(x*)| = {omega2}")
    print(f"Bound for |R_2(x*)| ≤ (M / 6) * |ω_3(x*)| = {r2_bound} * M \n")

    print(f"Value of L_3(x*) = {p3_star}")
    print(f"Check: L_3({x_check}) = {p3_check} (expected: {y_check})")
    print(f"|ω_4(x*)| = {omega3}")
    print(f"Bound for |R_3(x*)| ≤ (M / 24) * |ω_4(x*)| = {r3_bound} * M")

    draw(x_quad, y_quad, x_cub, y_cub, x_star, p2_star, p3_star, omega2, omega3, r2_bound, r3_bound)


if __name__ == "__main__":
    main()
