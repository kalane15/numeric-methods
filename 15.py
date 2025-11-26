from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ode_rhs(x: float, y: float) -> float:
    """Правая часть исходного ОДУ."""
    return (math.sqrt(y**2 + x**2) - y) / -x


def analytic_solution(x: float) -> float:
    """Аналитическое решение задачи Коши."""
    return x * math.sinh(math.log(1/x))


Tableau = Dict[str, Any]


EXPLICIT_TABLEAUS: List[Tableau] = [
    {
        "name": "Эйлер (1 порядок)",
        "order": 1,
        "c": (0.0,),
        "a": ((0.0,),),
        "b": (1.0,),
        "implicit": False,
    },
    {
        "name": "Эйлер-Коши (2 порядок)",
        "order": 2,
        "c": (0.0, 1.0),
        "a": ((0.0, 0.0), (1.0, 0.0)),
        "b": (0.5, 0.5),
        "implicit": False,
    },
    {
        "name": "Классический РК4",
        "order": 4,
        "c": (0.0, 0.5, 0.5, 1.0),
        "a": (
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.5, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        "b": (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        "implicit": False,
    },
]


IMPLICIT_TABLEAUS: List[Tableau] = [
    {
        "name": "Неявный Эйлер",
        "order": 1,
        "c": (1.0,),
        "a": ((1.0,),),
        "b": (1.0,),
        "implicit": True,
    },
    {
        "name": "Метод трапеций (Лобатто IIa)",
        "order": 2,
        "c": (0.0, 1.0),
        "a": ((0.0, 0.0), (0.5, 0.5)),
        "b": (0.5, 0.5),
        "implicit": True,
    },
    {
        "name": "Гаусс 4-го порядка",
        "order": 4,
        "c": (
            0.5 - math.sqrt(3.0) / 6.0,
            0.5 + math.sqrt(3.0) / 6.0,
        ),
        "a": (
            (
                0.25,
                0.25 - math.sqrt(3.0) / 6.0,
            ),
            (
                0.25 + math.sqrt(3.0) / 6.0,
                0.25,
            ),
        ),
        "b": (0.5, 0.5),
        "implicit": True,
    },
]


def build_grid(a: float, b: float, h: float) -> List[float]:
    """Создаёт равномерную сетку, гарантируя включение правого конца."""
    xs: List[float] = []
    n = 0
    while True:
        x = a + n * h
        if x > b + 1e-12:
            break
        xs.append(x)
        n += 1
    if abs(xs[-1] - b) > 1e-12:
        xs.append(b)
    return xs


def max_norm(values: Iterable[float]) -> float:
    return max(abs(v) for v in values)


def solve_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Прямой Гаусс с частичным выбором главного элемента."""
    n = len(rhs)
    a = [list(row) for row in matrix]
    b = list(rhs)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda i: abs(a[i][col]))
        if abs(a[pivot_row][col]) < 1e-15:
            raise RuntimeError("Матрица плохо обусловлена")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]
        pivot = a[col][col]
        inv_pivot = 1.0 / pivot
        for j in range(col, n):
            a[col][j] *= inv_pivot
        b[col] *= inv_pivot
        for row in range(col + 1, n):
            factor = a[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))
    return x


def numerical_jacobian(
    func: Callable[[Sequence[float]], List[float]],
    point: Sequence[float],
    base_value: Sequence[float] | None = None,
) -> List[List[float]]:
    s = len(point)
    jac = [[0.0 for _ in range(s)] for _ in range(s)]
    if base_value is None:
        base_value = func(point)
    for j in range(s):
        step = 1e-8 * max(1.0, abs(point[j]))
        perturbed = list(point)
        perturbed[j] += step
        values = func(perturbed)
        for i in range(s):
            jac[i][j] = (values[i] - base_value[i]) / step
    return jac


def fixed_point_iteration(
    tableau: Tableau,
    h: float,
    x_cur: float,
    y_cur: float,
    initial: Sequence[float],
    rhs: Callable[[float, float], float],
) -> List[float]:
    """Простая итеративная поправка на случай проблем с Ньютоном."""
    s = len(tableau["c"])
    stages = list(initial)
    for _ in range(25):
        new_stages: List[float] = []
        for i in range(s):
            arg_y = y_cur
            for j in range(s):
                arg_y += h * tableau["a"][i][j] * stages[j]
            new_stages.append(rhs(x_cur + tableau["c"][i] * h, arg_y))
        delta = max(abs(ns - os) for ns, os in zip(new_stages, stages))
        stages = new_stages
        if delta < 1e-12:
            break
    return stages


def solve_stages(
    tableau: Tableau,
    h: float,
    x_cur: float,
    y_cur: float,
    explicit: bool,
    rhs: Callable[[float, float], float],
) -> List[float]:
    s = len(tableau["c"])
    stages = [0.0 for _ in range(s)]
    if explicit:
        for i in range(s):
            arg_y = y_cur
            for j in range(i):
                arg_y += h * tableau["a"][i][j] * stages[j]
            stages[i] = rhs(x_cur + tableau["c"][i] * h, arg_y)
        return stages

    def residual(values: Sequence[float]) -> List[float]:
        res: List[float] = []
        for i in range(s):
            arg_y = y_cur
            for j in range(s):
                arg_y += h * tableau["a"][i][j] * values[j]
            res.append(values[i] - rhs(x_cur + tableau["c"][i] * h, arg_y))
        return res

    guess = [rhs(x_cur + c_i * h, y_cur) for c_i in tableau["c"]]
    stages = guess[:]
    for _ in range(12):
        res = residual(stages)
        if max_norm(res) < 1e-12:
            return stages
        jac = numerical_jacobian(residual, stages, res)
        delta = solve_linear_system(jac, [-value for value in res])
        stages = [k + dk for k, dk in zip(stages, delta)]
        if max(abs(dk) for dk in delta) < 1e-12:
            return stages
    return fixed_point_iteration(tableau, h, x_cur, y_cur, stages, rhs)


def rk_step(
    tableau: Tableau,
    h: float,
    x_cur: float,
    y_cur: float,
    rhs: Callable[[float, float], float],
) -> float:
    explicit = not tableau["implicit"]
    stages = solve_stages(tableau, h, x_cur, y_cur, explicit, rhs)
    increment = sum(b_i * k_i for b_i, k_i in zip(tableau["b"], stages))
    return y_cur + h * increment


def solve_cauchy(
    tableau: Tableau,
    x0: float,
    x_end: float,
    y0: float,
    h: float,
    rhs: Callable[[float, float], float],
) -> Tuple[List[float], List[float]]:
    xs = build_grid(x0, x_end, h)
    ys = [y0]
    for idx in range(len(xs) - 1):
        x_cur = xs[idx]
        y_cur = ys[-1]
        y_next = rk_step(tableau, h, x_cur, y_cur, rhs)
        ys.append(y_next)
    return xs, ys


def collect_results(
    tableaus: Sequence[Tableau],
    h: float,
    x0: float,
    x_end: float,
    y0: float,
    rhs: Callable[[float, float], float],
) -> List[dict]:
    results = []
    for tableau in tableaus:
        xs, ys = solve_cauchy(tableau, x0, x_end, y0, h, rhs)
        exact = [analytic_solution(x) for x in xs]
        errors = [abs(y - y_exact) for y, y_exact in zip(ys, exact)]
        results.append(
            {
                "tableau": tableau,
                "xs": xs,
                "ys": ys,
                "exact": exact,
                "errors": errors,
                "max_error": max(errors),
                "mean_error": sum(errors) / len(errors),
                "final_error": errors[-1],
            }
        )
    return results


def print_summary(title: str, results: Sequence[dict]) -> None:
    print(f"\n{title}")
    header = f"{'Метод':35s} {'p':>3s} {'max|ош|':>12s} {'ср.ош':>12s} {'|ош| в x=5':>12s}"
    print(header)
    print("-" * len(header))
    for res in results:
        t = res["tableau"]
        print(
            f"{t['name']:35s} {t['order']:3d} "
            f"{res['max_error']:12.5e} "
            f"{res['mean_error']:12.5e} "
            f"{res['final_error']:12.5e}"
        )


def plot_results(explicit_results: Sequence[dict], implicit_results: Sequence[dict]) -> None:
    smooth_x = np.arange(1.0, 5.0 + 1e-9, 0.01)
    smooth_y = [analytic_solution(val) for val in smooth_x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, res_set, title in zip(
        axes,
        (explicit_results, implicit_results),
        ("Явные схемы", "Неявные схемы"),
    ):
        ax.plot(smooth_x, smooth_y, color="black", label="Аналитика")
        for res in res_set:
            ax.plot(
                res["xs"],
                res["ys"],
                marker="o",
                linestyle="--",
                label=res["tableau"]["name"],
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)

    fig.suptitle("Сравнение результатов решения задачи Коши")
    plt.tight_layout()
    plt.show()


def main() -> None:
    x0 = 0.4
    x_end = 3.0
    y0 = 0.42
    h = 0.2

    explicit_results = collect_results(EXPLICIT_TABLEAUS, h, x0, x_end, y0, ode_rhs)
    implicit_results = collect_results(IMPLICIT_TABLEAUS, h, x0, x_end, y0, ode_rhs)

    print("Численное решение задачи Коши y' = x(x+2)y^3 + (x+3)y^2, y(1) = -2/3")
    print(f"Шаг сетки h = {h:.2f}, отрезок [{x0}; {x_end}]")
    print_summary("Явные методы", explicit_results)
    print_summary("Неявные методы", implicit_results)
    plot_results(explicit_results, implicit_results)


if __name__ == "__main__":
    main()

