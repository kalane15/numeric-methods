import math
import sys

import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def rhs(x: float, y: float, v: float) -> tuple[float, float]:
    """Правая часть системы: y' = v, v' = f(x, y, v)."""
    dy = v
    dv = (3 * x * x * v - (3 * x + 2) * y) / (x**4)
    return dy, dv


def rk4_step(x: float, y: float, v: float, h: float) -> tuple[float, float]:
    """Шаг метода Рунге–Кутты 4-го порядка для системы из двух уравнений."""
    k1_y, k1_v = rhs(x, y, v)
    k2_y, k2_v = rhs(x + h / 2, y + h * k1_y / 2, v + h * k1_v / 2)
    k3_y, k3_v = rhs(x + h / 2, y + h * k2_y / 2, v + h * k2_v / 2)
    k4_y, k4_v = rhs(x + h, y + h * k3_y, v + h * k3_v)

    y_next = y + (h / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    v_next = v + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return y_next, v_next


def integrate(
    alpha: float, x0: float, y0: float, h: float, steps: int
) -> tuple[list[float], list[float], list[float]]:
    """Интегрирует задачу Коши при заданном начальном наклоне alpha."""
    xs = [x0]
    ys = [y0]
    vs = [alpha]
    x = x0
    y = y0
    v = alpha

    for _ in range(steps):
        y, v = rk4_step(x, y, v, h)
        x = round(x + h, 12)  # устраняем накопление ошибок сложения
        xs.append(x)
        ys.append(y)
        vs.append(v)

    return xs, ys, vs


def shooting(
    x0: float,
    xb: float,
    y0: float,
    yb: float,
    h: float,
    alpha0: float,
    alpha1: float,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> tuple[list[float], list[float], list[float], float, list[dict]]:
    """Метод стрельбы с использованием метода секущих."""
    steps = int(round((xb - x0) / h))
    history: list[dict] = []

    xs_prev, ys_prev, vs_prev = integrate(alpha0, x0, y0, h, steps)
    res_prev = ys_prev[-1] - yb
    history.append(
        {"iter": 0, "alpha": alpha0, "y_end": ys_prev[-1], "residual": res_prev}
    )

    xs_curr, ys_curr, vs_curr = integrate(alpha1, x0, y0, h, steps)
    res_curr = ys_curr[-1] - yb
    history.append(
        {"iter": 1, "alpha": alpha1, "y_end": ys_curr[-1], "residual": res_curr}
    )

    iter_idx = 1
    while abs(res_curr) > tol and iter_idx < max_iter:
        denom = res_curr - res_prev
        if abs(denom) < 1e-14:
            raise RuntimeError("Метод секущих остановлен из-за малого знаменателя.")

        alpha_next = alpha1 - res_curr * (alpha1 - alpha0) / denom

        alpha0, res_prev = alpha1, res_curr
        xs_prev, ys_prev, vs_prev = xs_curr, ys_curr, vs_curr

        alpha1 = alpha_next
        xs_curr, ys_curr, vs_curr = integrate(alpha1, x0, y0, h, steps)
        res_curr = ys_curr[-1] - yb

        iter_idx += 1
        history.append(
            {
                "iter": iter_idx,
                "alpha": alpha1,
                "y_end": ys_curr[-1],
                "residual": res_curr,
            }
        )

    if abs(res_curr) > tol:
        raise RuntimeError(
            "Метод стрельбы не достиг требуемой точности за допустимое число итераций."
        )

    return xs_curr, ys_curr, vs_curr, alpha1, history


def fmt(value: float, width: int = 12, precision: int = 6) -> str:
    """Форматирует число без экспоненциальной записи."""
    return f"{value:{width}.{precision}f}"


def exact_solution(x: float) -> float:
    """Аналитическое решение из условия задачи."""
    return x * math.exp(-2 / x) + x * math.exp(-1 / x)


def plot_solutions(xs: list[float], ys: list[float], x_start: float, x_end: float) -> None:
    """Строит графики численного и точного решений."""
    dense_x = np.arange(x_start, x_end + 0.01, 0.01)
    dense_y = [exact_solution(x) for x in dense_x]

    plt.figure(figsize=(8, 5))
    plt.plot(dense_x, dense_y, label="Аналитическое решение", linewidth=2)
    plt.plot(xs, ys, "o-", label="Численное решение (РК4)", markersize=4)

    plt.title("Сравнение численного и аналитического решений")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    x_start = 0.5
    x_end = 5.0
    y_start = 0.076
    y_end = 7.445
    h = 0.5

    alpha_guess_0 = 1.0
    alpha_guess_1 = 1.6

    xs, ys, vs, alpha, history = shooting(
        x_start,
        x_end,
        y_start,
        y_end,
        h,
        alpha_guess_0,
        alpha_guess_1,
        tol=1e-9,
    )

    print("Решаем краевую задачу методом стрельбы:")
    print("x^4 * y'' - 3x^2 * y' + (3x + 2) * y = 0")
    print(f"y({x_start}) = {y_start}, y({x_end}) = {y_end}, шаг h = {h}\n")

    print("Итерации метода секущих для подбора начального наклона y'(0.5):")
    print(f"{'k':>2} {'alpha':>12} {'y(b)':>12} {'невязка':>12}")
    for item in history:
        print(
            f"{item['iter']:>2} "
            f"{fmt(item['alpha'], 12, 8)} "
            f"{fmt(item['y_end'], 12, 8)} "
            f"{fmt(item['residual'], 12, 6)}"
        )

    print(f"\nПодобранный наклон y'({x_start}) = {fmt(alpha, precision=8)}\n")

    print("Сравнение численного решения с аналитическим:")
    print(f"{'i':>2} {'x':>6} {'y числ.':>12} {'y точн.':>12} {'ошибка':>12}")
    max_error = 0.0
    for i, (x, y_num) in enumerate(zip(xs, ys)):
        y_true = exact_solution(x)
        error = abs(y_true - y_num)
        max_error = max(max_error, error)
        print(
            f"{i:>2} {x:>6.2f} "
            f"{fmt(y_num, 12, 8)} "
            f"{fmt(y_true, 12, 8)} "
            f"{fmt(error, 12, 6)}"
        )

    print(
        f"\nМаксимальная абсолютная погрешность на сетке: {fmt(max_error, precision=6)}"
    )

    plot_solutions(xs, ys, x_start, x_end)


if __name__ == "__main__":
    main()
