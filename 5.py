A = [[15, 6, -4, -3, 89],
     [-4, 13, -2, 5, 61],
     [6, -5, -19, 7, 46],
     [-5, -8, 4, 18, 68]
     ]

tol = 0.0001


def check(A) -> bool:
    n = len(A)
    for i in range(n):
        mx = abs(A[i][i])
        sm_row = 0
        sm_str = 0
        for j in range(n):
            if i != j:
                sm_row += abs(A[i][j])
                sm_str += abs(A[j][i])
        if mx < sm_row and mx < sm_str:
            return False

    return True


def solve(A):
    n = len(A)
    x = [A[i][n] for i in range(n)]

    for i in range(n):
        if A[i][i] == 0:
            raise ValueError(f"A[i][i] = 0 error")

    for iteration in range(10000000):
        max_diff = 0.0

        for i in range(n):
            s = 0.0
            for j in range(i):
                s += A[i][j] * x[j]
            for j in range(i + 1, n):
                s += A[i][j] * x[j]

            new_val = (A[i][n] - s) / A[i][i]
            diff = abs(new_val - x[i])
            if diff > max_diff:
                max_diff = diff
            x[i] = new_val

        if max_diff < tol:
            print(f"Solution has been found after {iteration + 1} iteration.")
            return x

    return x


def check_solution(A, x):
    n = len(A)

    max_error = 0.0
    print("computed  |  expected |  error")
    for i in range(n):
        computed = sum(A[i][j] * x[j] for j in range(n))
        expected = A[i][n]
        error = abs(computed - expected)

        if error > max_error:
            max_error = error

        print(f"{computed:9.6f} | {expected:9.1f} | {error:9.6f}")

    print(f"Max error: {max_error:.6f}")


def main():
    if not check(A):
        print("Matrix is incorrect")

    x = solve(A)
    print("X:", x)

    check_solution(A, x)


if __name__ == "__main__":
    main()
