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


def get_newton_error(x, x_star, degree):
    err = 1
    for i in range(degree + 1):
        err *= abs(x_star - x[i])
    return err


def build_newton(x_t, y_t, x_star, start, stop, degree):
    x_quad = x_t[start:stop]
    y_quad = y_t[start:stop]

    x_check = x_quad[1]
    y_check = y_quad[1]

    dd = divided_differences(x_quad, y_quad)
    p2_star = newton_eval(x_quad, dd, x_star, degree)
    p2_check = newton_eval(x_quad, dd, x_check, degree)

    x = np.linspace(x_t[0], x_t[-1], 100)
    y = []
    for i in x:
        y.append(newton_eval(x_quad, dd, i, degree))

    err = get_newton_error(x_quad, x_star, degree)
    print(f"Value of L_{degree}(x*) = {p2_star}")
    print(f"Check: L_{degree}({x_check}) = {p2_check} (expected: {y_check})")
    print(f"Approximate error  = {err}")

    return x, y, err, p2_star, x_quad, y_quad


def build_newton_of_degree(x_all, y_all, x_star, degree):
    error_to_value = dict()
    data = []
    inital_index = 0
    for i in range(len(x_all)):
        if x_star > x_all[i]:
            inital_index = i
            break

    for i in range(0, degree):
        x, y, err, p_star, points_x, points_y = build_newton(x_all, y_all, x_star, inital_index + i,
                                                             inital_index + i + degree + 1, degree)
        error_to_value[err] = p_star
        data.append((f"Newton degree {degree}", x, y, x_star, p_star, points_x, points_y))

    plot_multiple_graphs(data)

    best = error_to_value[min(error_to_value.keys())]

    print(f"Answer with min err is {best}")


def main():
    x_all = x_table.copy()
    y_all = y_table.copy()

    x_star = 4.016

    build_newton_of_degree(x_all, y_all, x_star, 2)
    build_newton_of_degree(x_all, y_all, x_star, 3)


if __name__ == "__main__":
    main()
