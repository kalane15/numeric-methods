import matplotlib.pyplot as plt

def solve_tdma(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d.
    a - lower diagonal (a[0] is unused)
    b - main diagonal
    c - upper diagonal (c[n-1] is unused)
    d - right-hand side (vector)
    """
    N = len(d)
    c_prime = [0.0] * N
    d_prime = [0.0] * N
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, N):
        temp = b[i] - a[i] * c_prime[i-1]
        if i < N - 1:
            c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / temp
        
    # Back substitution
    x = [0.0] * N
    x[N-1] = d_prime[N-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

def draw(plot_x_list, plot_y_list, x, y, x_star, y_star, target_i):
    plt.figure(figsize=(12, 7))
    plt.plot(plot_x_list, plot_y_list, label="Natural Cubic Spline S(x)", color="blue")
    plt.scatter(x, y, color="red", zorder=5, label="Interpolation Nodes (x_i, y_i)")
    if target_i != -1:
        plt.scatter([x_star], [y_star], color="green", zorder=6, s=100, 
                    edgecolors="black", label=f"Point x* = {x_star}, S(x*) = {y_star:.4f}")

    plt.title("Natural Cubic Spline")
    plt.xlabel("x")
    plt.ylabel("y = S(x)")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

def main():
    x = [-2.64, -1.639, -0.352, 1.507, 2.794, 3.938, 5.654, 6.941, 8.085, 9.944, 11.66]
    y = [0.642, 0.847, 0.219, 0.368, 0.046, -0.152, 0.461, 0.985, 0.939, 1.362, 1.167]
    x_star = 9.094
    n = len(x) - 1 # number of segments

    # --- Step 1: Calculate step sizes h_i ---
    # Using 1-based indexing for h, A, B, C, D for convenience
    # h[i] = x_i - x_{i-1}
    h = [0.0] * (n + 1)
    for i in range(1, n + 1):
        h[i] = x[i] - x[i-1]

    # --- Step 2: Form the SLAE for c_i ---
    # We are solving the system for c_2, ..., c_n (n-1 unknowns)
    # c_1 = 0 by the natural spline condition 
    N_system = n - 1 # System size

    a_tdma = [0.0] * N_system # Lower diagonal
    b_tdma = [0.0] * N_system # Main diagonal
    c_tdma = [0.0] * N_system # Upper diagonal
    d_tdma = [0.0] * N_system # Right-hand side

    # Calculate the RHS: D_i = 3 * ( (y_i - y_{i-1})/h_i - (y_{i-1} - y_{i-2})/h_{i-1} )
    RHS = [0.0] * (n + 1)
    for i in range(2, n + 1):
        term1 = (y[i] - y[i-1]) / h[i]
        term2 = (y[i-1] - y[i-2]) / h[i-1]
        RHS[i] = 3 * (term1 - term2)

    # Filling A and d matrices for TDMA
    for i_tdma in range(N_system):
        i = i_tdma + 2
        
        # Main diagonal: 2 * (h_{i-1} + h_i)
        b_tdma[i_tdma] = 2 * (h[i-1] + h[i])
        
        # Right-hand side
        d_tdma[i_tdma] = RHS[i]
        
        # Lower diagonal: h_{i-1}
        if i_tdma > 0: # a_tdma[0] is unused
            a_tdma[i_tdma] = h[i-1]
            
        # Upper diagonal: h_i
        if i_tdma < N_system - 1: # c_tdma[N_system-1] is unused
            c_tdma[i_tdma] = h[i]

    # --- Step 3: Solve the SLAE ---
    # c_solved contains [c_2, c_3, ..., c_n]
    c_solved = solve_tdma(a_tdma, b_tdma, c_tdma, d_tdma)

    # Forming the full C array (1-based index)
    C = [0.0] * (n + 1)
    C[1] = 0.0 # c_1 = 0
    for i in range(N_system):
        C[i+2] = c_solved[i]
    # Now C = [0.0, c_1, c_2, ..., c_n]

    # --- Step 4: Calculate coefficients A, B, D ---
    A = [0.0] * (n + 1)
    B = [0.0] * (n + 1)
    D = [0.0] * (n + 1)

    for i in range(1, n + 1):
        # a_i = y_{i-1} 
        A[i] = y[i-1]
        
        if i < n:
            # d_i = (c_{i+1} - c_i) / (3 * h_i) 
            D[i] = (C[i+1] - C[i]) / (3.0 * h[i])
            # b_i = (y_i - y_{i-1})/h_i - h_i/3 * (c_{i+1} + 2*c_i) 
            B[i] = (y[i] - y[i-1]) / h[i] - (h[i] / 3.0) * (C[i+1] + 2.0 * C[i])
        else: # i == n (last segment)
            # d_n = -c_n / (3 * h_n)
            D[n] = -C[n] / (3.0 * h[n])
            # b_n = (y_n - y_{n-1})/h_n - 2*h_n/3 * c_n
            B[n] = (y[n] - y[n-1]) / h[n] - (2.0 * h[n] / 3.0) * C[n]

    # --- Step 5: Calculate S(x*) and find coefficients ---
    # Find the segment i to which x* belongs
    target_i = -1
    y_star = 0.0
    for i in range(1, n + 1):
        if x[i-1] <= x_star <= (x[i]):
            target_i = i
            break


    # Theoretical error factor
    # |R(x)| <= (5/384) * H^4 * max|f^(4)(x)|
    h = [0.0] * (n + 1)
    H = 0.0
    for i in range(1, n + 1):
        h[i] = x[i] - x[i-1]
        if h[i] > H:
            H = h[i]
    H_power_4 = H**4
    max_error_factor = (5.0 / 384.0) * H_power_4

    if target_i != -1:
        # Calculate S(x*)
        dx = x_star - x[target_i - 1]
        y_star = (A[target_i] + 
                  B[target_i] * dx + 
                  C[target_i] * (dx**2) + 
                  D[target_i] * (dx**3))
        
        print(f"Error analysis:")
        print(f" Maximum grid pitch H = {H:.4f}")
        print(f" H^4 = {H_power_4:.5f}")
        print(f" Theoretical margin of error: |R(x)| <= {max_error_factor:.5f} * max|f^(4)(x)|\n")
        
        print(f"Point x* = {x_star} is on segment i = {target_i} (between x_{target_i-1} and x_{target_i}).")
        print(f"  x_{target_i-1} = {x[target_i-1]}, x_{target_i} = {x[target_i]}\n")
        
        print(f"Spline Value at x*:")
        print(f"  S({x_star}) = {y_star:.7f}\n")
        
        print(f"Spline Coefficients on segment i = {target_i}:")
        print(f"  a_{target_i} = {A[target_i]:.7f}")
        print(f"  b_{target_i} = {B[target_i]:.7f}")
        print(f"  c_{target_i} = {C[target_i]:.7f}")
        print(f"  d_{target_i} = {D[target_i]:.7f}")

    plot_x_list = []
    plot_y_list = []

    for i in range(1, n + 1):
        x_start = x[i-1]
        x_end = x[i]
        
        num_points = 200
        
        local_x = []
        if num_points == 1:
            local_x.append(x_start)
        else:
            for j in range(num_points):
                local_x.append(x_start + (x_end - x_start) * j / (num_points - 1))
        
        if i > 1:
            local_x = local_x[1:]
            
        for val in local_x:
            dx = val - x[i-1]
            s_val = (A[i] + B[i]*dx + C[i]*(dx**2) + D[i]*(dx**3))
            plot_x_list.append(val)
            plot_y_list.append(s_val)

    draw(plot_x_list, plot_y_list, x, y, x_star, y_star, target_i)

if __name__ == "__main__":
    main()