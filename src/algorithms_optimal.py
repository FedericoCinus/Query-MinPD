import cvxpy as cp
import numpy as np

# Optimal
def cvx_optimizer(A_eq: np.matrix, 
                  s: np.matrix,
                  max_iters: int = None, 
                  eps: float = None, 
                  solver: str = 'SCS', 
                  verbosity = 0):
    # Ensure A_eq is symmetric
    assert np.allclose(A_eq, A_eq.T), "A_eq must be symmetric"

    # Define the degree matrix D
    D = np.diag(np.sum(A_eq, axis=0))

    # Define the initial Laplacian matrix L
    L_initial = D - A_eq

    # Define the size of the problem
    n = len(s)

    # Define the optimization variable L
    L = cp.Variable((n, n), symmetric=True, integer=False)

    # Define the identity matrix I
    I = np.eye(n)

    # Define the constraints
    constraints = [
        cp.trace(L) == np.trace(L_initial),  # budget constraint
        L @ np.ones(n) == np.zeros(n) # laplacian matrix
    ]
    # 1. No new edges constraint
    constraints += [L[i, j] == 0. for i in range(n) for j in range(i + 1, n) if L_initial[i, j] == 0.]
    
    # 2. Keep signs
    constraints += [L[i, i] >= 0. for i in range(n)]  # Diagonal elements non-negative
    constraints += [L[i, j] <= 0. for i in range(n) for j in range(n) if i != j]  # Off-diagonal elements non-positive


    # Additional constraints to ensure off-diagonal elements are non-positive
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(L[i, j] <= 0)

    # Define the objective function
    objective = cp.Minimize(cp.matrix_frac(s, I + L))

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Define solver parameters
    solver_params = {}
    if max_iters is not None:
        solver_params['max_iters'] = max_iters
    if eps is not None:
        solver_params['eps'] = eps

    # Solve the problem
    prob.solve(solver=solver)#, verbose=verbosity, **solver_params)

    # Convert the optimized Laplacian matrix L to the adjacency matrix A
    L_opt = L.value
    D_opt = np.diag(np.diagonal(L_opt))
    A_opt = D_opt - L_opt
    
    ε = 1E-4
    A_opt[np.absolute(A_opt) < ε] = 0

    # Collect statistics
    stats = {
        'solve_time': prob.solver_stats.solve_time,
        'setup_time': prob.solver_stats.setup_time,
        'status': prob.status
    }
    return A_opt, [], stats