import math
import sys

import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def rhs(x: float, y: float, v: float) -> tuple[float, float]:
    dy = v
    dv = (2.0 * (x * math.tan(x) + 1.0) * v - 2.0 * y * math.tan(x)) / x
    return dy, dv


def rk4_step(x: float, y: float, v: float, h: float) -> tuple[float, float]:
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
    xs = [x0]
    ys = [y0]
    vs = [alpha]
    x = x0
    y = y0
    v = alpha

    for _ in range(steps):
        y, v = rk4_step(x, y, v, h)
        x = round(x + h, 12)
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
    steps = int(round((xb - x0) / h))
    history: list[dict] = []

    xs_prev, ys_prev, _ = integrate(alpha0, x0, y0, h, steps)
    res_prev = ys_prev[-1] - yb
    history.append(
        {
            "iter": 0,
            "alpha": alpha0,
            "y_end": ys_prev[-1],
            "residual": res_prev,
            "xs": xs_prev,
            "ys": ys_prev,
        }
    )

    xs_curr, ys_curr, vs_curr = integrate(alpha1, x0, y0, h, steps)
    res_curr = ys_curr[-1] - yb
    history.append(
        {
            "iter": 1,
            "alpha": alpha1,
            "y_end": ys_curr[-1],
            "residual": res_curr,
            "xs": xs_curr,
            "ys": ys_curr,
        }
    )

    iter_idx = 1
    while abs(res_curr) > tol and iter_idx < max_iter:
        denom = res_curr - res_prev
        if abs(denom) < 1e-14:
            raise RuntimeError("Метод секущих остановлен из-за малого знаменателя.")

        alpha_next = alpha1 - res_curr * (alpha1 - alpha0) / denom

        alpha0, res_prev = alpha1, res_curr
        xs_prev, ys_prev = xs_curr, ys_curr

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
                "xs": xs_curr,
                "ys": ys_curr,
            }
        )

    if abs(res_curr) > tol:
        raise RuntimeError(
            "Метод стрельбы не достиг требуемой точности за допустимое число итераций."
        )

    return xs_curr, ys_curr, vs_curr, alpha1, history


def fmt(value: float, width: int = 12, precision: int = 6) -> str:
    return f"{value:{width}.{precision}f}"


def exact_solution(x: float) -> float:
    return x * math.tan(x) + math.tan(x) - x + 1


def plot_solutions(
        xs: list[float], ys: list[float], x_start: float, x_end: float
) -> None:
    dense_x = np.arange(x_start, x_end + 0.01, 0.01)
    dense_y = [exact_solution(x) for x in dense_x]

    plt.figure(figsize=(8, 5))
    plt.plot(dense_x, dense_y, label="Аналитическое решение", linewidth=2)
    plt.plot(xs, ys, "o-", label="Численное решение", markersize=4)

    plt.title("Сравнение численного и аналитического решений")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_shooting_iterations(history: list[dict], x_end: float, y_end: float) -> None:
    plt.figure(figsize=(8, 5))
    overshoot_color = "tab:red"
    undershoot_color = "tab:blue"
    exact_color = "tab:green"

    for item in history:
        xs = item["xs"]
        ys = item["ys"]
        residual = item["residual"]

        if abs(residual) < 1e-10:
            verdict = "точное попадание"
            color = exact_color
            linestyle = "-"
        elif residual > 0:
            verdict = "перелёт"
            color = overshoot_color
            linestyle = "--"
        else:
            verdict = "недолёт"
            color = undershoot_color
            linestyle = "--"

        plt.plot(
            xs,
            ys,
            label=f"k={item['iter']} ({verdict})",
            color=color,
            linestyle=linestyle,
        )

    plt.scatter([x_end], [y_end], color="black", marker="x", s=80, label="Цель y(b)")
    plt.title("Итерации метода секущих (стрельба)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    x_start = 2.0
    x_end = 4.2
    y_start = -7.555
    y_end = 6.044
    h = 0.11

    alpha_guess_0 = 4
    alpha_guess_1 = 8

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

    plot_shooting_iterations(history, x_end, y_end)
    plot_solutions(xs, ys, x_start, x_end)


if __name__ == "__main__":
    main()
