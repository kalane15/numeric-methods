import matplotlib.pyplot as plt
import math


def draw(x_data, y_data, x_star, results, m_values):
    plt.figure(figsize=(10, 6))

    plt.scatter(x_data, y_data, color='red', label='Original Data $y_i$', marker='o')

    x_range = linspace(min(x_data), max(x_data), 100)

    colors = {1: 'blue', 2: 'green', 3: 'orange'}

    for m in m_values:
        coeffs = results[m]['coeffs']
        Fm_x_range = [polynomial_value(coeffs, x) for x in x_range] 
        
        plt.plot(x_range, Fm_x_range, color=colors[m], linestyle='-', 
                 label=f'$F_{m}(x)$ (degree {m})')
        
        plt.scatter(x_star, results[m]['Fm_x_star'], color=colors[m], marker='*', s=150)

    plt.scatter([], [], color='purple', 
                 label=f'$x^*$ = {x_star}', marker='x', s=100)

    plt.title('Approximation using Least Squares Method')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True)
    plt.legend()
    plt.show()


def linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def gauss_solve(A_in, B_in):
    """
    Solves a system of linear equations A * X = B using the Gaussian elimination method.
    """
    A = [row[:] for row in A_in]
    B = B_in[:]
    n = len(A)

    # 1. Create the augmented matrix M = [A | B]
    M = []
    for i in range(n):
        M.append(A[i] + [B[i]])

    # 2. Forward Elimination
    for i in range(n):
        # Partial Pivoting: Find the row with the largest absolute value for the pivot element
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        
        # Swap rows
        M[i], M[max_row] = M[max_row], M[i]

        # Check for singularity
        if M[i][i] == 0:
            raise ValueError("Matrix is singular or the system has no unique solution.")

        # Eliminate all entries below the pivot M[i][i]
        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]

    # 3. Backward Substitution
    X = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_of_products = 0.0
        for j in range(i + 1, n):
            sum_of_products += M[i][j] * X[j]
            
        X[i] = (M[i][n] - sum_of_products) / M[i][i]

    return X


def calculate_sums(m, x, y):
    """
    Calculates the necessary sums for the LSM Normal System of degree m.
    """
    n = len(x)
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    B = [0.0] * (m + 1)

    # Calculate elements of matrix A: A[k][i] = Sum(x^(i+k))
    for k in range(m + 1):
        for i in range(m + 1):
            power = i + k
            sum_val = 0.0
            for val in x:
                sum_val += math.pow(val, power)
            A[k][i] = sum_val

    # Calculate elements of vector B: B[k] = Sum(y * x^k)
    for k in range(m + 1): # Row index
        power_x = k
        sum_val = 0.0
        for idx in range(n):
            sum_val += y[idx] * math.pow(x[idx], power_x)
        B[k] = sum_val

    return A, B


def polynomial_value(coeffs, x_val):
    """
    Calculates the polynomial value F_m(x) = sum(a_i * x^i) at a given point.
    """
    value = 0.0
    for i, a in enumerate(coeffs):
        value += a * math.pow(x_val, i)
    return value


def calculate_errors(coeffs, m, x, y, N_points):
    """
    Calculates the Sum of Squared Errors (Phi_m) and the Standard Error of the Estimate (Sigma_m).
    """
    # Calculate F_m(x_i) values
    Fm_x = [polynomial_value(coeffs, x_val) for x_val in x]
    
    # Sum of Squared Errors Phi_m = Sum((Fm_x_i - y_i)^2)
    Phi_m = 0.0
    for i in range(N_points):
        Phi_m += (Fm_x[i] - y[i]) ** 2
    
    # Standard Error of the Estimate Sigma_m = sqrt(Phi_m / (N - (m+1)))
    degrees_of_freedom = N_points - (m + 1)
    if degrees_of_freedom <= 0:
        Sigma_m = float('nan')
    else:
        Sigma_m = math.sqrt(Phi_m / degrees_of_freedom)
        
    return Phi_m, Sigma_m


def main():
    x_data = [-3.54, -2.83, -2.12, -1.41, -0.70, 0.01, 0.72, 1.43, 2.14, 2.85, 3.56]
    y_data = [-0.3467, 0.9794, 1.7685, 2.0287, 1.9168, 1.2173, 0.4136, 0.0678, 0.2081, 0.8348, 1.1254]
    x_star = 1.468
    N_points = len(x_data)

    print(f"Number of points (N): {N_points}")
    print(f"Point x* for calculation: {x_star}\n")

    results = {}
    m_values = [1, 2, 3]

    for m in m_values:
        print(f"--- Polynomial of degree m = {m} ---")
        
        # 4.1. Building the Normal System
        A, B = calculate_sums(m, x_data, y_data)
        
        print("Normal LSM System (A|B):")
        for i in range(m + 1):
            equation = ""
            for j in range(m + 1):
                equation += f"{A[i][j]:.4f}*a{j} "
                if j < m:
                    equation += "+ "
            equation += f"= {B[i]:.4f}"
            print(equation)

        print()
        
        # 4.2. Solving the Normal System
        a_coeffs = gauss_solve(A, B)
        
        # 4.3. Forming the Approximating Polynomial
        polynomial_str = f"F_{m}(x) = "
        for i, a in enumerate(a_coeffs):
            if i == 0:
                polynomial_str += f"{a:.4f}"
            elif i == 1:
                # Use just x for x^1
                term = f"{abs(a):.4f}x"
                polynomial_str += f" + {term}" if a >= 0 else f" - {term}"
            else:
                term = f"{abs(a):.4f}x^{i}"
                polynomial_str += f" + {term}" if a >= 0 else f" - {term}"
        
        # 4.4. Calculating Errors
        Phi_m, Sigma_m = calculate_errors(a_coeffs, m, x_data, y_data, N_points)
        
        # 4.5. Calculating value at x*
        Fm_x_star = polynomial_value(a_coeffs, x_star)
        
        results[m] = {
            'coeffs': a_coeffs,
            'Phi_m': Phi_m,
            'Sigma_m': Sigma_m,
            'Fm_x_star': Fm_x_star,
            'polynomial_str': polynomial_str.replace('+ -', '- ')
        }

    print("{:<10} {:<24} {:<15} {:<25}".format("Sequence", "Sum of Squared Errors", "ASD", "Value in x*"))
    for m in m_values:
        print("{:<10} {:<24.4f} {:<15.4f} {:<25.4f}".format(m, results[m]['Phi_m'], results[m]['Sigma_m'], results[m]['Fm_x_star']))

    print("\nApproximating polynomials:")
    for m in m_values:
        print(f"m={m}: {results[m]['polynomial_str']}")

    draw(x_data, y_data, x_star, results, m_values)


if __name__ == "__main__":
    main()