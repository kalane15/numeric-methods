import math
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

A = 0.5  # левая граница
B = 6.5  # правая граница
H = 0.5  # шаг сети
LEFT_BC = 0.515
RIGHT_BC = 14.738


def a(x: float):
    return x * x * (3 * x + 2)


def p(x: float) -> float:
    return (-2.0 * x * (3 * x + 4)) / a(x)


def q(x: float) -> float:
    return 6 * (x + 2) / a(x)


def f(x: float) -> float:
    return 0.0


def analytic_solution(x: float) -> float:
    return (x * x * (1 + x)) / (3 * x + 2)


def build_system(order: int) -> Tuple[List[List[float]], List[float]]:
    n_intervals = round((B - A) / H)  # количество интервалов
    size = n_intervals + 1  # количество узлов
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    # Левая граница: y(a) + y'(a) = LEFT_BC
    if order == 1:
        # Односторонняя разность для y'(a)
        matrix[0][0] = 1.0 - 1.0 / H  # для y(a)
        matrix[0][1] = 1.0 / H  # для y_1 (разность для y')
        rhs[0] = LEFT_BC
    elif order == 2:
        # Второй порядок аппроксимации для y'(a)
        matrix[0][0] = 1.0 - 3.0 / (2 * H)
        matrix[0][1] = 4.0 / (2 * H)
        matrix[0][2] = -1.0 / (2 * H)
        rhs[0] = LEFT_BC
    else:
        raise ValueError("Допустимые порядки аппроксимации: 1 или 2.")

    # Основная часть для внутренних точек
    for k in range(1, size - 1):
        xk = A + k * H
        matrix[k][k - 1] = 1.0 / (H * H) - p(xk) / (2.0 * H)
        matrix[k][k] = -2.0 / (H * H) + q(xk)
        matrix[k][k + 1] = 1.0 / (H * H) + p(xk) / (2.0 * H)
        rhs[k] = f(xk)

    # Правая граница: y(b) = RIGHT_BC
    matrix[-1][-1] = 1.0  # только для y(b)
    rhs[-1] = RIGHT_BC

    return matrix, rhs


def gaussian_elimination(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    b = rhs[:]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("Обнаружен нулевой поворотный элемент.")
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]

        pivot_val = a[i][i]
        for j in range(i, n):
            a[i][j] /= pivot_val
        b[i] /= pivot_val

        for r in range(i + 1, n):
            factor = a[r][i]
            if factor == 0.0:
                continue
            for j in range(i, n):
                a[r][j] -= factor * a[i][j]
            b[r] -= factor * b[i]

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))
    return x


def solve_bvp(order: int) -> List[float]:
    matrix, rhs = build_system(order)
    return gaussian_elimination(matrix, rhs)


def print_table(xs: List[float], exact: List[float], sol1: List[float], sol2: List[float]) -> None:
    header = (
        f"{'k':>3} {'x':>8} {'y_точно':>15} "
        f"{'y_1-й П.':>15} {'|ε_1|':>15} "
        f"{'y_2-й П.':>15} {'|ε_2|':>15}"
    )
    print(header)
    print("-" * len(header))
    for i, x in enumerate(xs):
        err1 = abs(sol1[i] - exact[i])
        err2 = abs(sol2[i] - exact[i])
        print(
            f"{i:3d} {x:8.3f} {exact[i]:15.6f} "
            f"{sol1[i]:15.6f} {err1:15.6f} "
            f"{sol2[i]:15.6f} {err2:15.6f}"
        )


def plot_solutions(xs: List[float], exact: List[float], sol1: List[float], sol2: List[float]) -> None:
    dense_steps = 400
    xs_dense = [A + i * (B - A) / dense_steps for i in range(dense_steps + 1)]
    ys_dense = [analytic_solution(x) for x in xs_dense]

    plt.figure(figsize=(10, 6))
    plt.plot(xs_dense, ys_dense, label="Аналитическое решение", color="black")
    plt.plot(xs, sol1, marker="o", label="ГУ: 1-й порядок")
    plt.plot(xs, sol2, marker="s", label="ГУ: 2-й порядок")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Сравнение аналитического и численных решений")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    n_intervals = round((B - A) / H)
    xs = [A + i * H for i in range(n_intervals + 1)]
    exact_values = [analytic_solution(x) for x in xs]

    solution_order1 = solve_bvp(1)
    solution_order2 = solve_bvp(2)

    print("Краевая задача: 4x^2 y'' + 8xy' - (4x^2 + 12x + 3)y = 0")
    print(f"Интервал: [{A}; {B}], шаг h = {H}")
    print(f"Граничные условия: y'({A}) = {LEFT_BC}, y'({B}) = {RIGHT_BC}\n")

    print_table(xs, exact_values, solution_order1, solution_order2)

    err1 = max(abs(solution_order1[i] - exact_values[i]) for i in range(len(xs)))
    err2 = max(abs(solution_order2[i] - exact_values[i]) for i in range(len(xs)))
    print(f"\nМаксимальная абсолютная погрешность (ГУ 1-го порядка): {err1:.6f}")
    print(f"Максимальная абсолютная погрешность (ГУ 2-го порядка): {err2:.6f}")

    plot_solutions(xs, exact_values, solution_order1, solution_order2)


if __name__ == "__main__":
    main()
