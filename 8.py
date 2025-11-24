import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
from math import e


def draw():
    x1_range = np.linspace(-3, 3, 400)
    x2_range = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    F1 = np.zeros_like(X1)
    F2 = np.zeros_like(X2)

    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            F1[j, i] = f1(x1_range[i], x2_range[j])
            F2[j, i] = f2(x1_range[i], x2_range[j])

    plt.figure()
    plt.contour(X1, X2, F1, [0], colors='red')
    plt.contour(X1, X2, F2, [0], colors='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.title('Graphs of the equations to find initial approximations')
    plt.show()


def f1(x1, x2):
    return e ** (x1 * x2) + x1 - 4


def f2(x1, x2):
    return x1 ** 2 - 4 * x2 ** 2 + 1


def df1_dx1(x1, x2):
    return x2 * e ** (x1 * x2) + 1


def df1_dx2(x1, x2):
    return x1 * e ** (x1 * x2)


def df2_dx1(x1, x2):
    return 2 * x1


def df2_dx2(x1, x2):
    return -8 * x2


LAMBDA1 = 0
LAMBDA2 = 0


def phi1(x1, x2):
    return x1 + LAMBDA1 * f1(x1, x2)


def phi2(x1, x2):
    return x2 + LAMBDA2 * f2(x1, x2)


def dphi1_dx1(x1, x2):
    return 1 + LAMBDA1 * df1_dx1(x1, x2)


def dphi1_dx2(x1, x2):
    return LAMBDA1 * df1_dx2(x1, x2)


def dphi2_dx1(x1, x2):
    return LAMBDA2 * df2_dx1(x1, x2)


def dphi2_dx2(x1, x2):
    return 1 + LAMBDA2 * df2_dx2(x1, x2)


def check_convergence(x1, x2):
    a, b, c, d = (dphi1_dx1(x1, x2),
                  dphi1_dx2(x1, x2),
                  dphi2_dx1(x1, x2),
                  dphi2_dx2(x1, x2))
    # print(a, b, c, d)
    q1 = abs(a) + abs(b)
    q2 = abs(c) + abs(d)

    q = max(q1, q2)

    if int(q) == 1:
        print("q == 1, сходимость не гарантирована")
        return True

    return q < 1


MAX_ITERS = 1000


def simple_iteration(x1, x2, eps=1e-4):
    if not check_convergence(x1, x2):
        raise Exception("Метод неприменим")

    it = 0
    while True:
        it += 1
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1, x2)

        if max(abs(x1 - x1_new), abs(x2 - x2_new)) <= eps:
            return x1_new, x2_new, it
        x1 = x1_new
        x2 = x2_new

        if it > MAX_ITERS:
            raise Exception(f"Метод не сошелся за {it} итераций")


def seidel_method(x1, x2, eps=1e-4):
    if not check_convergence(x1, x2):
        raise Exception("Метод неприменим")

    it = 0
    while True:
        it += 1
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1_new, x2)

        if max(abs(x1 - x1_new), abs(x2 - x2_new)) <= eps:
            return x1_new, x2_new, it
        x1 = x1_new
        x2 = x2_new

        if it > MAX_ITERS:
            raise Exception(f"Метод не сошелся за {it} итераций")


def newton_method(x1, x2, eps=1e-4):
    J = [[df1_dx1(x1, x2), df1_dx2(x1, x2)],
         [df2_dx1(x1, x2), df2_dx2(x1, x2)]]

    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    if abs(det) == 0:
        raise Exception("Матрица Якоби вырождена!")

    it = 0
    x1_pred = x1
    x2_pred = x2
    while True:
        it += 1
        ff1 = f1(x1, x2)
        ff2 = f2(x1, x2)

        dx = (ff1 * J[1][1] - ff2 * J[0][1]) / det
        dy = (J[0][0] * ff2 - J[1][0] * ff1) / det
        x1 -= dx
        x2 -= dy
        if max(abs(x1 - x1_pred), abs(x2 - x2_pred)) <= eps:
            return x1, x2, it
        x1_pred = x1
        x2_pred = x2

        if it > MAX_ITERS:
            raise Exception(f"Метод не сошелся за {it} итераций")


def main():
    initials = [
        {"x1": -1.75, "x2": -1, "label": "Первый корень", "l1" : 0.02, "l2" : -0.1},
        {"x1": 1.248, "x2": 0.83, "label": "Второй корень", "l1" : -0.05, "l2" : 0.1}
    ]
    global LAMBDA1, LAMBDA2

    # Run and print results
    for init in initials:
        print(f"\n--- {init['label']} ---")
        print("Приближение: x1={:.4f}, x2={:.4f}".format(init['x1'], init['x2']))
        LAMBDA1 = init["l1"]
        LAMBDA2 = init["l2"]
        # Simple Iteration
        x1, x2, it = simple_iteration(init['x1'], init['x2'])
        print(f"Простой итерации: x1={x1 :.6f}, x2={x2 :.6f}, итераций={it}")
        print(f"f1={round(f1(x1, x2), 2)}, f2={round(f2(x1, x2), 2)}\n")

        # Seidel method
        x1, x2, it = seidel_method(init['x1'], init['x2'])
        print(f"Метод зейделя: x1={x1:.6f}, x2={x2:.6f}, итераций={it}")
        print(f"f1={round(f1(x1, x2), 2)}, f2={round(f2(x1, x2), 2)}\n")

        # Newton's method
        x1, x2, it_n = newton_method(init['x1'], init['x2'])
        print("Метод Ньютона: x1={:.6f}, x2={:.6f}, итераций={}".format(x1, x2, it_n))
        print(f"f1={round(f1(x1, x2), 2)}, f2={round(f2(x1, x2), 2)}\n")

    return True


if __name__ == "__main__":
    # draw()
    main()
