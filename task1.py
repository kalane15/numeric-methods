from copy import deepcopy


def read_linear_system_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {filename} не найден")

    A = []

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        line = line.replace(',', ' ')

        coefficients = line.split()

        try:
            coeffs = [float(x) for x in coefficients]

            A.append(coeffs)

        except ValueError:
            raise ValueError(f"Некорректные данные в уравнении {i}: '{line}'")

    return A


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

def print_matrix_e(m):
    for i in m:
        print(*[round(j, 1) for j in i])
    print()

def to_step_matrix(coeffs, reversed):
    det = 1

    for i in range(len(coeffs)):
        tmp = coeffs[i][i]
        det *= coeffs[i][i]
        for j in range(len(coeffs[i])):
            coeffs[i][j] /= tmp
            if j < len(coeffs[i]) - 1:
                reversed[i][j] /= tmp
        #
        for n in range(i + 1, len(coeffs)):
            tmp2 = coeffs[n][i]
            for j in range(len(coeffs[n])):
                coeffs[n][j] -= coeffs[i][j] * tmp2
                if j < len(coeffs[n]) - 1:
                    reversed[n][j] -= reversed[i][j] * tmp2
        # print_matrix(coeffs)
        # print_matrix(reversed_matrix)
    return round(det, 3)


def gauss_back(coeffs, reversed):
    n = len(coeffs)
    res = []
    for _ in range(n):
        res.append(0)

    for k in range(n - 1, -1, -1):
        res[k] = coeffs[k][-1]

        for j in range(k, n - 1):
            res[k] -= coeffs[k][j + 1] * res[j + 1]

            for m in range(len(reversed[k])):
                reversed[k][m] -= coeffs[k][j + 1] * reversed[j + 1][m]

    for i in range(len(reversed)):
        for j in range(len(reversed[i])):
            reversed[i][j] = round(reversed[i][j], 4)

    res = [round(i, 3) for i in res]
    return res


def main():
    A = read_linear_system_from_file("t1.txt")
    # print(A)

    A_inital = deepcopy(A)
    if A[0][0] == 0:
        print("Не выполнены условия, а11 = 0")
        return

    reversed_matrix = []

    for i in range(len(A)):
        reversed_matrix.append([int(j == i) for j in range(len(A))])

    det = to_step_matrix(A, reversed_matrix)

    if det == 0:
        print("Не выполнены условия, det A = 0")
        return

    X = gauss_back(A, reversed_matrix)
    print("X:", X)
    print("Определитель:", det)
    print("Обратная матрица")
    print_matrix(reversed_matrix)

    AA = [s[:-1] for s in A_inital ]

    b = [s[-1] for s in A_inital ]

    X = [[i] for i in X]  # [a, b, c] -> [[a], [b], [c]] массив из N элементов в матрицу Nx1

    out = matrix_multiply(AA, X)

    print(X)
    print("Проверка")
    s1 = "Коэффициенты, полученны подстановкой"
    s2 = "Входные коэффициенты"
    print(s1, s2)

    for i in range(len(out)):
        print(str(out[i][0]).ljust(len(s1), " "), b[i])

    print("Проверка обратной матрицы")
    A_no_b = [i[:-1] for i in A_inital]
    print_matrix_e(matrix_multiply(A_no_b, reversed_matrix))


if __name__ == "__main__":
    main()
