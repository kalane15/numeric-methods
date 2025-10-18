from typing import List
import numpy as np

Matrix = List[List[float]]
Vector = List[float]


def print_vector(v):
    tmp = [round(i, 4) for i in v]
    print(*tmp)
    print()


def multiply_matrix_and_vector(A: Matrix, v: Vector) -> Vector:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError(f"Несовместимые размеры матриц: A[{len(A)}x{len(A[0])}] * B[{len(B)}x{len(B[0])}]")
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    result = [[0.0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result


def print_matrix(m):
    for i in m:
        print(*[round(j, 4) for j in i])
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
    return get_det == 0


def decompose_to_lu(A: Matrix, b: Vector) -> Matrix:
    n = len(A)
    lu_matrix = [[0.0 for _ in range(2 * n + 1)] for _ in range(n)]

    for i in range(n):
        lu_matrix[i][i] = 1.0

    for k in range(n):
        for j in range(k, n):
            sum_row = sum(lu_matrix[k][i] * lu_matrix[i][j + n] for i in range(k))
            lu_matrix[k][j + n] = A[k][j] - sum_row

        for i in range(k + 1, n):
            sum_col = sum(lu_matrix[i][m] * lu_matrix[m][k + n] for m in range(k))
            lu_matrix[i][k] = (A[i][k] - sum_col) / lu_matrix[k][k + n]

    for i in range(n):
        lu_matrix[i][2 * n] = b[i]

    return lu_matrix


def forward_solve_l(aug: Matrix) -> Vector:
    n = len(aug)
    y = [0.0] * n
    for i in range(n):
        # Use L matrix (columns 0 to n-1)
        sum_lower = sum(aug[i][j] * y[j] for j in range(i))
        y[i] = aug[i][2 * n] - sum_lower
    return y


def back_solve_u(aug: Matrix, y: Vector) -> Vector:
    n = len(aug)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # Use U matrix (columns n to 2n-1)
        sum_upper = sum(aug[i][j + n] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_upper) / aug[i][i + n]
    return x


def det(aug: Matrix) -> float:
    determinant = 1.0
    n = len(aug)
    # Use diagonal elements of U (columns n to 2n-1)
    for i in range(n):
        determinant *= aug[i][i + n]
    return determinant


def compute_inverse(aug: Matrix) -> Matrix:
    n = len(aug)
    inverse = [[0.0 for _ in range(n)] for _ in range(n)]
    for k in range(n):
        rhs = [1.0 if i == k else 0.0 for i in range(n)]
        aug_k = [aug[i][:2 * n] + [rhs[i]] for i in range(n)]
        y = forward_solve_l(aug_k)
        x = back_solve_u(aug_k, y)
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

    aug = decompose_to_lu(A, b)
    print(f"Augmented LU matrix:")

    print_matrix(aug)

    y = forward_solve_l(aug)
    solution = back_solve_u(aug, y)
    print(f"Solution x:")
    print_vector(solution)

    Ax_vector = multiply_matrix_and_vector(A, solution)
    print(f"Check Ax:")
    print_vector(Ax_vector)

    determinant = det(aug)
    print(f"Determinant of matrix A: \n{determinant}")

    np_A = np.array(A)
    np_determinant = np.linalg.det(np_A)
    print(f"Check determinant: \n{np_determinant}")
    print()

    inverse_matrix = compute_inverse(aug)
    print(f"Inverse matrix A:")
    print_matrix(inverse_matrix)

    n = len(A)
    result = [[sum(A[i][k] * inverse_matrix[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    print(f"Check A * A^{-1}:")
    print_matrix(result)


if __name__ == "__main__":
    main()
