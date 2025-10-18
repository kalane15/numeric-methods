import copy
from math import sqrt
import cmath


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
    if not m:
        print()
        return

    # Находим максимальную ширину элемента во всей матрице
    max_width = 0
    for row in m:
        for element in row:
            # Преобразуем элемент в строку с округлением до 4 знаков
            element_str = str(round(element, 4))
            max_width = max(max_width, len(element_str))

    # Выводим матрицу с выравниванием по максимальной ширине
    for row in m:
        formatted_row = [f"{round(element, 4):>{max_width}}" for element in row]
        print(*formatted_row)
    print()


def transpose_matrix(matrix) -> list:
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def get_e_matrix(size) -> list:
    result_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        result_matrix[i][i] = 1

    return result_matrix


def matrux_multiply_number(number, matrix) -> list:
    if isinstance(number, list) and isinstance(matrix, int):
        matrix, number = number, matrix
    elif isinstance(number, int) and isinstance(matrix, list):
        pass
    else:
        print("Write matrix and number")
        return []

    result_matrix = copy.deepcopy(matrix)
    for i in range(len(result_matrix)):
        for j in range(len(result_matrix)):
            result_matrix[i][j] *= number

    return result_matrix


def divide_matrix_by_element(a, b) -> list:
    _answerMatrix = a.copy()
    for i in range(len(a)):
        for j in range(len(a[0])):
            if b[i][j] == 0:
                _answerMatrix[i][j] = 0
            else:
                _answerMatrix[i][j] /= b[i][j]
    return _answerMatrix


def new_matrix_from_digit(num, rows, cols):
    return [[num for _ in range(cols)] for _ in range(rows)]


def get2x2_eigenvalues(matrix) -> list:
    if len(matrix) != 2:
        raise ValueError("Matrix should be 2x2")

    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    discriminant = (a + d) ** 2 - 4 * (a * d - b * c)

    if discriminant > 0:
        return [((a + d) + sqrt(discriminant)) / 2, ((a + d) - sqrt(discriminant)) / 2]
    elif discriminant < 0:
        return [((a + d) + cmath.sqrt(discriminant)) / 2, ((a + d) - cmath.sqrt(discriminant)) / 2]
    else:
        return [(a + d) / 2, (a + d / 2)]


def piecemeal_subtraction(matrix_a, matrix_b) -> list:
    return [[matrix_a[i][j] - matrix_b[i][j] for j in range(len(matrix_a[0]))] for i in range(len(matrix_a))]


def extract_blocks(matrix, tol=1e-4):
    """
    Разбивает почти треугольную матрицу на блоки 1x1 и 2x2
    """
    n = len(matrix)
    blocks = []
    i = 0

    while i < n:
        if i == n - 1:
            blocks.append([[matrix[i][i]]])
            i += 1
        else:
            if abs(matrix[i + 1][i]) < tol:
                blocks.append([[matrix[i][i]]])
                i += 1
            else:
                # Блок 2x2
                block_2x2 = [
                    [matrix[i][i], matrix[i][i + 1]],
                    [matrix[i + 1][i], matrix[i + 1][i + 1]]
                ]
                blocks.append(block_2x2)
                i += 2

    return blocks


def GetQR(matrix):
    r = matrix.copy()
    n = len(matrix)
    q = get_e_matrix(n)
    for i in range(n - 1):
        vi = [0.0 for _ in range(i)]

        subcolumn_norm = sqrt(sum(matrix[j][i] ** 2 for j in range(i, n)))

        sign = 1 if matrix[i][i] >= 0 else -1
        first_element = matrix[i][i] + sign * subcolumn_norm
        vi.append(first_element)

        for j in range(i + 1, n):
            vi.append(matrix[j][i])

        vi = transpose_matrix([vi])

        # print(vi)
        vi1Matrix = matrix_multiply(vi, transpose_matrix(vi))
        vi2Matrix = matrix_multiply(transpose_matrix(vi), vi)

        # print(vi1Matrix)
        # print(vi2Matrix)

        vi2Matrix = new_matrix_from_digit(vi2Matrix[0][0], len(vi1Matrix), len(vi1Matrix))

        H = piecemeal_subtraction(get_e_matrix(len(vi2Matrix)),
                                  matrux_multiply_number(2, divide_matrix_by_element(vi1Matrix, vi2Matrix)))
        # print(H)
        r = matrix_multiply(H, r)
        q = matrix_multiply(q, H)
        matrix = r.copy()
    return q, r


new_matrix = [[1, -5, -4, 7, 3],
              [4, 12, 1, 2, 4],
              [-2, 3, 4, 7, 5],
              [2, 5, -4, 11, 3],
              [5, 4, -5, -4, -2]]

# new_matrix = [[5, -2, 1, 3],
#               [2, -4, 3, -1],
#               [-3, 3, 3, 5],
#               [1, 2, -5, 2]]
prev_l = []
eps = 1e-6
it = 0
while True:
    q, r = GetQR(new_matrix)
    new_matrix = matrix_multiply(r, q)
    blocks = extract_blocks(new_matrix, eps)
    l = []
    for block in blocks:
        if len(block) == 1:
            l.append(block[0][0])

        else:
            t1, t2 = get2x2_eigenvalues(block)
            l.append(t1)
            l.append(t2)

    if len(prev_l) == 0:
        prev_l = l.copy()
        continue

    mx_diff = 0.0
    for i in range(len(l)):
        # print(l[i])
        # print(prev_l[i])
        mx_diff = max(abs(prev_l[i] - l[i]), mx_diff)
    if mx_diff < eps:
        break
    else:
        prev_l = l.copy()
        it += 1

print(f"Решение найдено после {it} итераций")
print("Полученная матрица")
print_matrix(new_matrix)
print("Собственные значения:")
for i in l:
    print(i)
