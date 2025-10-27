import matplotlib.pyplot as plt
import numpy as np

x_table = [3.05, 3.43, 3.81, 4.19, 4.57, 4.95, 5.33, 5.71, 6.09]
y_table = [1.8571, 2.1247, 3.6456, 2.6842, 2.3539, 0.3431, 1.6577, 2.8982, 1.4326]


def draw(x_quad, y_quad, dd_quad, x_cub, y_cub, dd_cub, x_star, p2_star, p3_star, r2_approx, r3_approx):
    """
    Function for visualizing Newton interpolation results.
    Builds two plots: for P_2(x) and P_3(x), with table points,
    interpolation nodes, x*, and error estimation as text.
    """
    # Full table of points

    x_plot = np.linspace(0.5, 4.0, 200)

    # Compute polynomial values
    y_quad_plot = [newton_eval(x_quad, dd_quad, x, 2) for x in x_plot]
    y_cub_plot = [newton_eval(x_cub, dd_cub, x, 3) for x in x_plot]

    _, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(x_plot, y_quad_plot, 'b-', label='P₂(x)', linewidth=2)
    ax1.scatter(x_table, y_table, color='red', s=50, label='Table points', zorder=5)
    ax1.scatter(x_quad, y_quad, color='green', s=100, marker='o', label='Nodes for P₂', zorder=5)
    ax1.scatter(x_star, p2_star, color='purple', s=80, marker='*', label=f'x* = {x_star}, P₂(x*) ≈ {p2_star:.4f}',
                zorder=5)
    ax1.text(0.05, 0.95, f'Approx |R₂| ≈ {r2_approx:.6f}', transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Newton Interpolation Polynomial of 2nd Degree')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.show()

    _, ax3 = plt.subplots(1, 1, figsize=(10, 6))

    ax3.plot(x_plot, y_cub_plot, 'g-', label='P₃(x)', linewidth=2)
    ax3.scatter(x_table, y_table, color='red', s=50, label='Table points', zorder=5)
    ax3.scatter(x_cub, y_cub, color='orange', s=100, marker='s', label='Nodes for P₃', zorder=5)
    ax3.scatter(x_star, p3_star, color='purple', s=80, marker='*', label=f'x* = {x_star}, P₃(x*) ≈ {p3_star:.4f}',
                zorder=5)
    ax3.text(0.05, 0.95, f'Approx |R₃| ≈ {r3_approx:.6f}', transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Newton Interpolation Polynomial of 3rd Degree')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.show()


def divided_differences(x_vals, y_vals):
    """
    Computes the divided difference table.
    dd[k][j] = f[x_j, ..., x_{j+k}]
    """
    m = len(x_vals)
    dd = [[0.0] * m for _ in range(m)]
    for j in range(m):
        dd[0][j] = y_vals[j]
    for k in range(1, m):
        for j in range(m - k):
            dd[k][j] = (dd[k - 1][j + 1] - dd[k - 1][j]) / (x_vals[j + k] - x_vals[j])
    return dd


def newton_eval(x_vals, dd, x, degree):
    """
    Evaluates Newton's polynomial of given degree at x.
    P(x) = dd[0][0] + dd[1][0]*(x - x0) + dd[2][0]*(x - x0)(x - x1) + ...
    """
    result = dd[0][0]
    prod = 1.0
    for k in range(1, degree + 1):
        prod *= (x - x_vals[k - 1])
        result += dd[k][0] * prod
    return result


def main():
    x_all = x_table.copy()
    y_all = y_table.copy()

    x_star = 4.016
    x_check = x_table[2]
    y_check = y_table[2]

    # For second degree (n=2, points i=1,2,3)
    x_quad = x_all[1:4]
    y_quad = y_all[1:4]
    dd_quad = divided_differences(x_quad, y_quad)
    p2_star = newton_eval(x_quad, dd_quad, x_star, 2)
    p2_check = newton_eval(x_quad, dd_quad, x_check, 2)

    # For third degree (n=3, points i=1,2,3,4)
    x_cub = x_all[1:5]
    y_cub = y_all[1:5]
    dd_cub = divided_differences(x_cub, y_cub)
    p3_star = newton_eval(x_cub, dd_cub, x_star, 3)
    p3_check = newton_eval(x_cub, dd_cub, x_check, 3)

    # For error estimates differently: using P_{n+1}(x*) - P_n(x*)
    r2_approx = abs(p3_star - p2_star)

    # For P4 (degree 4, points i=1 to 5)
    x_err = x_all[1:6]
    y_err = y_all[1:6]
    dd_err = divided_differences(x_err, y_err)
    p4_star = newton_eval(x_err, dd_err, x_star, 4)

    r3_approx = abs(p4_star - p3_star)

    print(f"Value of P_2(x*) = {p2_star}")
    print(f"Check: P_2({x_check}) = {p2_check} (expected: {y_check})")
    print(f"Approximate error |R_2(x*)| ≈ {r2_approx}\n")

    print(f"Value of P_3(x*) = {p3_star}")
    print(f"Check: P_3({x_check}) = {p3_check} (expected: {y_check})")
    print(f"Approximate error |R_3(x*)| ≈ {r3_approx}")

    draw(x_quad, y_quad, dd_quad, x_cub, y_cub, dd_cub, x_star, p2_star, p3_star, r2_approx, r3_approx)


if __name__ == "__main__":
    main()
