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


def progonka(coeffs):
    prev_p = 0
    prev_q = 0
    p_list = []
    q_list = []
    det = 1
    for n in range(len(coeffs)):
        if n == (len(coeffs) - 1):
            a, b, c, d = coeffs[n][n - 1], coeffs[n][n], 0, coeffs[n][-1]
        elif n == 0:
            a, b, c, d = 0, coeffs[n][n], coeffs[n][n + 1], coeffs[n][-1]
        else:
            a, b, c, d = coeffs[n][n - 1], coeffs[n][n], coeffs[n][n + 1], coeffs[n][-1]

        p = -c / (b + a * prev_p)
        q = (d - a * prev_q) / (b + a * prev_p)
        det *= (b + a * prev_p)
        p_list.append(p)
        q_list.append(q)
        prev_q = q
        prev_p = p

    res = []
    for _ in range(len(coeffs)):
        res.append(0)

    res[-1] = q_list[-1]

    for n in range(len(coeffs) - 2, -1, -1):
        x = p_list[n] * res[n + 1] + q_list[n]
        res[n] = x

    return res, det


def check(coeffs):
    for n in range(len(coeffs)):
        if n == (len(coeffs) - 1):
            a, b, c, d = coeffs[n][n - 1], coeffs[n][n], 0, coeffs[n][-1]
        elif n == 0:
            a, b, c, d = 0, coeffs[n][n], coeffs[n][n + 1], coeffs[n][-1]
        else:
            a, b, c, d = coeffs[n][n - 1], coeffs[n][n], coeffs[n][n + 1], coeffs[n][-1]

        if n == 0:
            if abs(c / b) >= 1:
                return False

        if n == len(coeffs) - 1:
            if abs(a / b) >= 1:
                return False

        if abs(b) < abs(a) + abs(c):
            return False

    return True


if __name__ == "__main__":
    A = read_linear_system_from_file("t3.txt")

    if not check(A):
        print("Не выполнены условия для метода")

    X, det = progonka(A)
    X = [round(i, 3) for i in X]

    AA = [s[:-1] for s in A]

    b = [s[-1] for s in A]

    X = [[i] for i in X] # [a, b, c] -> [[a], [b], [c]] массив из N элементов в матрицу Nx1

    out = matrix_multiply(AA, X)
    print("Решение:")
    print(X)
    print()

    print("Определитель:", det)
    print()
    print("Проверка")
    s1 = "Коэффициенты, полученны подстановкой"
    s2 = "Входные коэффициенты"
    print(s1, s2)

    for i in range(len(out)):
        print(str(out[i][0]).ljust(len(s1), " "), b[i])
