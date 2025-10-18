from typing import List
import numpy as np

Matrix = List[List[float]]
Vector = List[float]


def multiply_matrix_and_vector(A: Matrix, v: Vector) -> Vector:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def print_matrix(m):
    for i in m:
        print(*[round(k, 5) for k in i])
    print()


def print_vector(m):
    print(*[round(k, 5) for k in m])
    print()


def get_det(matrix):
    n = len(matrix)

    for row in matrix:
        if len(row) != n:
            raise ValueError("Матрица должна быть квадратной")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        minor = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            minor.append(row)
        minor_det = get_det(minor)
        sign = 1 if j % 2 == 0 else -1
        det += sign * matrix[0][j] * minor_det

    return det


def is_determinant_zero(matrix: Matrix) -> bool:
    return get_det(matrix) == 0


def decompose_to_lu(A: Matrix, b: Vector) -> [Matrix, Matrix]:
    n = len(A)
    l = [[0.0 for _ in range(n)] for _ in range(n)]
    u = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        l[i][i] = 1.0

    for i in range(n):
        for j in range(n):
            if i <= j:
                sum_1 = sum(l[i][k] * u[k][j] for k in range(i))
                u[i][j] = (A[i][j] - sum_1)
            else:
                sum_2 = sum(l[i][k] * u[k][j] for k in range(j))
                l[i][j] = (A[i][j] - sum_2) / u[j][j]
    return l, u


def forward_solve_l(aug: Matrix, b: Vector) -> Vector:
    n = len(aug)
    y = [0.0] * n
    for i in range(n):
        # Use L matrix (columns 0 to n-1)
        sum_lower = sum(aug[i][j] * y[j] for j in range(i))
        y[i] = b[i] - sum_lower
    return y


def back_solve_u(aug: Matrix, y: Vector) -> Vector:
    n = len(aug)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # Use U matrix (columns n to 2n-1)
        sum_upper = sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_upper) / aug[i][i]
    return x


def det(u: Matrix, l: Matrix) -> float:
    determinant = 1.0
    n = len(u)
    d1 = 1.0
    d2 = 1.0
    for i in range(n):
        d1 *= l[i][i]
        d2 *= u[i][i]
    return d1 * d2


def compute_inverse(u: Matrix, l: Matrix, b: Vector) -> Matrix:
    n = len(l)
    inverse = [[0.0 for _ in range(n)] for _ in range(n)]
    for k in range(n):
        rhs = [1.0 if i == k else 0.0 for i in range(n)]
        y = forward_solve_l(l, rhs)
        x = back_solve_u(u, y)
        for i in range(n):
            inverse[i][k] = x[i]
    return inverse


def main():
    A = [
        [6, 3, 4, 7, 1],
        [3, 2, 1, -5, 2],
        [7, 0, 5, 2, -7],
        [2, 5, -4, 1, 3],
        [3, 4, -5, 2, 5]
    ]
    b = [63, 69, 30, 13, 16]

    if is_determinant_zero(A):
        print("Matrix is singular, determinant is zero.")
        return

    l, u = decompose_to_lu(A, b)
    print(f"L and U matrixes")
    print_matrix(l)
    print_matrix(u)

    y = forward_solve_l(l, b)
    solution = back_solve_u(u, y)
    print(f"Solution x:")
    print_vector(solution)

    Ax_vector = multiply_matrix_and_vector(A, solution)
    print(f"Check Ax:")
    print_vector(Ax_vector)

    determinant = det(u, l)
    print(f"Determinant of matrix A: \n{round(determinant, 4)} \n")

    np_A = np.array(A)
    np_determinant = np.linalg.det(np_A)
    print(f"Check determinant: \n{round(np_determinant, 4)} \n")

    inverse_matrix = compute_inverse(u, l, b)
    print(f"Inverse matrix A:")
    print_matrix(inverse_matrix)

    n = len(A)
    result = [[sum(A[i][k] * inverse_matrix[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    print(f"Check A * A^{-1}: \n")
    print_matrix(result)


if __name__ == "__main__":
    main()
