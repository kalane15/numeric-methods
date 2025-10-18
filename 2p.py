from typing import List, Tuple

Matrix = List[List[float]]
Vector = List[float]


def CreateExtendedMatrix(A: Matrix, b: Vector) -> Matrix:
    n = len(A)
    extended = [[0.0 for _ in range(2 * n + 1)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            extended[i][j] = A[i][j]

    for i in range(n):
        extended[i][n] = b[i]

    for i in range(n):
        extended[i][n + 1 + i] = 1.0

    return extended


def ProcessLU(extended_matrix: Matrix) -> Matrix:
    n = len(extended_matrix)
    total_cols = len(extended_matrix[0])  # 2*n + 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(extended_matrix[i][i]) < 1e-15:
                raise ZeroDivisionError("Матрица вырождена")

            factor = extended_matrix[j][i] / extended_matrix[i][i]
            extended_matrix[j][i] = factor

            for k in range(i + 1, total_cols):
                extended_matrix[j][k] -= factor * extended_matrix[i][k]

    return extended_matrix


def SolveBackwardSubstitution(extended_matrix: Matrix) -> Matrix:
    n = len(extended_matrix)
    total_cols = len(extended_matrix[0])
    for i in range(n - 1, -1, -1):
        diag_element = extended_matrix[i][i]

        for j in range(i, total_cols):
            extended_matrix[i][j] /= diag_element

        for k in range(i):
            factor = extended_matrix[k][i]
            for j in range(i, total_cols):
                extended_matrix[k][j] -= factor * extended_matrix[i][j]

    return extended_matrix


def ExtractResults(extended_matrix: Matrix) -> Tuple[Vector, Matrix]:
    n = len(extended_matrix)
    solution = [extended_matrix[i][n] for i in range(n)]
    inverse_matrix = [[extended_matrix[i][n + 1 + j] for j in range(n)] for i in range(n)]

    return solution, inverse_matrix


def MultyplyMatrixAndVector(A: Matrix, v: Vector) -> Vector:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def PrintMatrix(M: Matrix) -> str:
    return "\n".join(["[ " + "  ".join(f"{round(val, 4)}" for val in row) + " ]" for row in M])


def PrintVector(v: Vector) -> str:
    return "[ " + "  ".join(f"{val:8.4g}" for val in v) + " ]"


def Det(extended_matrix: Matrix) -> float:
    det = 1.0
    for i in range(len(extended_matrix)):
        det *= extended_matrix[i][i]
    return det


def main():
    A = [
        [6.0, 3.0, 4.0, 7.0, 1.0],
        [3.0, 2.0, 1.0, -5.0, 2.0],
        [7.0, 0.0, 5.0, 2.0, -7.0],
        [2.0, 5.0, -4.0, 1.0, 3.0],
        [3.0, 4.0, -5.0, 2.0, 5.0]
    ]
    b = [63.0, 69.0, 30.0, 13.0, 16.0]

    print("Исходная матрица A:")
    print(PrintMatrix(A))
    print("\nВектор b:")
    print(PrintVector(b))
    print("")

    extended_matrix = CreateExtendedMatrix(A, b)
    print("Начальная расширенная матрица [A | b | I]:")
    print(PrintMatrix(extended_matrix))
    print("")

    extended_matrix = ProcessLU(extended_matrix)
    print("Матрица после LU-разложения:")
    print(PrintMatrix(extended_matrix))
    print("")

    print("Определитель A:")
    print(Det(extended_matrix))
    print("")

    extended_matrix = SolveBackwardSubstitution(extended_matrix)
    print("Матрица после обратной подстановки:")
    print(PrintMatrix(extended_matrix))
    print("")

    solution, inverse_a = ExtractResults(extended_matrix)
    print("Решение СЛАУ x:")
    print(PrintVector(solution))
    print("")

    Ax_vector = MultyplyMatrixAndVector(A, solution)
    print("Проверка ответа Ax:")
    print(PrintVector(Ax_vector))
    print("")

    print("Обратная матрица А:")
    print(PrintMatrix(inverse_a))
    print("")

    print("Проверка A * A^(-1):")
    result = [[sum(A[i][k] * inverse_a[k][j] for k in range(len(A))) for j in range(len(A))] for i in range(len(A))]
    print(PrintMatrix(result))


if __name__ == "__main__":
    main()