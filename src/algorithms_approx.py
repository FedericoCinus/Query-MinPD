import copy
import cvxpy as cp
import numpy as np
import sys
sys.path.append('../src')
import scipy
from scipy.sparse import identity
from sklearn.preprocessing import normalize
import time



def gradient_dense(X, s, obj_name="PD", is_directed=True):
    """Returns gradient data matrix
    s: np.matrix : column vector of the inner opinions
    X: np.matrix : adj matrix or laplacian matrix of the graph (directed or undirected)
    Is: np.array : row indices
    Js: np.array : column indices
    obj_name: str : (PD: polarization + disagreement, P: polarization, D: disagreement)
    is_directed: bool : directed or undirected graph
    """
    assert obj_name in ("PD", "P", "D"), f"{obj_name} not defined"
    if is_directed:
        return gradient_dense_directed(X, s, obj_name)
    else:
        return gradient_dense_undirected(X, s, obj_name)

def gradient_dense_directed(A, s, obj_name="PD"):
    """Returns gradient data matrix
    s: np.matrix : column vector of the inner opinions
    A: np.matrix : adj matrix of the graph (directed or undirected)
    obj_name: str : (PD: polarization + disagreement, P: polarization, D: disagreement)
    is_directed: bool : directed or undirected graph
    """
    if obj_name == "PD": # Polarization + Disagreement
        dim = A.shape
        A_rows = dim[0]
        A_cols = dim[1]
        dim = s.shape
        s_rows = dim[0]

        T_0 = (2 * np.eye(s_rows, s_rows))
        T_1 = np.linalg.inv((T_0 + -A.T))
        t_2 = (np.linalg.inv((T_0 - A))).dot(s)
        t_3 = (T_1).dot(t_2)
        T_4 = (2 * A)
        t_5 = (T_1).dot((((np.eye(s_rows, s_rows) + np.diag((np.ones(s_rows)).dot(A))) - T_4)).dot(t_2))
        t_6 = (s).dot(T_1)
        functionValue = ((s).dot(t_3) + (0.5 * (s).dot(t_5)))
        gradient = (((((2 * np.outer(t_3, t_6)) + (0.5 * np.outer(t_5, t_6))) + (0.5 * np.outer(np.ones(A_rows), (t_6 * t_6)))) - np.outer(t_2, t_6)) + (0.5 * np.outer((T_1).dot((((np.eye(s_rows, s_rows) + np.diag((A.T).dot(np.ones(s_rows)))) + -T_4.T)).dot(t_2)), t_6)))

        return functionValue, gradient

    
    elif obj_name == "P": # Polarization
        dim = A.shape
        A_rows = dim[0]
        A_cols = dim[1]
        dim = s.shape
        s_rows = dim[0]

        T_0 = (2 * np.eye(A_cols, s_rows))
        T_1 = np.linalg.inv((T_0 + -A.T))
        t_2 = (T_1).dot((np.linalg.inv((T_0 - A))).dot(s))
        functionValue = (s).dot(t_2)
        gradient = (2 * np.outer(t_2, (s).dot(T_1)))

        return functionValue, gradient

    elif obj_name == "D": # Disagreement
        dim = A.shape
        A_rows = dim[0]
        A_cols = dim[1]
        dim = s.shape
        s_rows = dim[0]

        T_0 = (2 * np.eye(A_cols, s_rows))
        T_1 = np.linalg.inv((T_0 + -A.T))
        T_2 = (2 * A)
        t_3 = (np.linalg.inv((T_0 - A))).dot(s)
        t_4 = (T_1).dot((((np.eye(s_rows, s_rows) + np.diag((np.ones(A_cols)).dot(A))) - T_2)).dot(t_3))
        t_5 = (s).dot(T_1)
        functionValue = (0.5 * (s).dot(t_4))
        gradient = ((((0.5 * np.outer(t_4, t_5)) + (0.5 * np.outer(np.ones(A_rows), (t_5 * t_5)))) - np.outer(t_3, t_5)) + (0.5 * np.outer((T_1).dot((((np.eye(s_rows, s_rows) + np.diag((A.T).dot(np.ones(A_cols)))) + -T_2.T)).dot(t_3)), t_5)))

        return functionValue, gradient

def gradient_dense_undirected(A, s, obj_name="PD"):
    """Returns gradient data matrix and objective
    s: np.matrix : column vector of the inner opinions
    A: np.matrix : laplacian matrix of the graph (directed or undirected)
    obj_name: str : (PD: polarization + disagreement, P: polarization, D: disagreement)
    is_directed: bool : directed or undirected graph

    Refer to: https://www.matrixcalculus.org/
    """ 
    if obj_name == "PD": # Polarization + Disagreement
        D = np.diag(np.sum(A, axis=0))
        L = D - A
        I = np.eye(A.shape[0])
        
        I_plus_L_inv = np.linalg.inv(I + L)
        functionValue = s.T @ I_plus_L_inv @ s #function value s^T (I + L)^{-1} s

        return float(functionValue), None
    else:
        raise Exception("Only PD objective/gradient are implemented.")

def GD_optimizer(X_initial, s: np.matrix,
                 max_iters: int, early_stopping: bool = False,
                 routine: str = None, grad_params: dict = None, lr_params: dict = None, 
                 verbosity: int = 0, ε = 1e-6, obj_name="PD", is_directed=True):
    """Returns the X optimized matrix and the objective values
       using input matrix A, inner opinion vector s, the percentage of budget to use,
       routine = "ADAM" / "simple"

       X --> Laplacian if undirected (Projection not implemented!)
       X --> Adjacency if directed

       Returns always adj matrix: X_inner, feasible_objectives, stats(None)
    """
    # -------------------------
    # 0. Params and stats
    setup_time = time.time()
    stats = {'increments': [], 'solve_time': None, 'setup_time': None}
    η = lr_params["lr_coeff"] # Learning rate
    Δlimit = 0.2 #edges*scale (scale=1e-5) # minimum detected decrement in objective
    

    # -------------------------
    # 1. Initialize variables    
    X0 = copy.deepcopy(X_initial)
    X_inner = copy.deepcopy(X_initial)
    X_prev = copy.deepcopy(X_inner)


    # -------------------------
    # 2. Define the projection constraints
    if not is_directed:
        raise Exception("Projection for undirected graph not defined")
    
    # -------------------------
    # 3. Initialization objective
    obj, _ = gradient_dense(np.asarray(X_inner), s.flatten(), obj_name=obj_name, is_directed=is_directed)
    feasible_objectives = [obj]
    X_increment = np.inf
    
    


    # -------------------------
    # 4. Optimization loop
    time0 = time.time()
    stats['setup_time'] = time.time() - setup_time
    T = 1
    
    print(f"  Initializing PGD {routine} solver with η={η}")
    for t in range(1, int(max_iters)+1):
        print(f"{t}/{max_iters+1}", end="\r")
        ### 4.1 Gradient step
        _, grad = gradient_dense(np.asarray(X_inner), s.flatten(), obj_name=obj_name, is_directed=is_directed)
        X_prev = copy.deepcopy(X_inner)
        if verbosity >= 1:
            if t%T == 0 or t == 1:
                Δ_obj = (np.min(feasible_objectives[-4:-3])/feasible_objectives[0] - np.min(feasible_objectives[-2:-1])/feasible_objectives[0]) if t > 5 else 0
                print(f"  t={t}, obj%={obj/feasible_objectives[0]*100:.3f}    Δ_obj%={Δ_obj:.3f}  min detectible Δ={Δlimit:.3f}")
              
        if routine == "ADAM": ## GRADIENT SCHEMES
            β1, β2, ε = (0.9, 0.999, 1e-08)
            μ = np.multiply(β1, μ) + np.multiply((1-β1), grad) if t>1 else np.multiply((1-β1), grad)
            v = np.multiply(β2, v) + np.multiply((1-β2), np.power(grad, 2)) if t>1 else np.multiply((1-β2), np.power(grad, 2))
            μ_hat = np.multiply(1 / (1 - np.power(β1, t)), μ)
            v_hat = np.multiply(1 / (1 - np.power(β2, t)), v)
            X_inner -= np.multiply(η/(np.sqrt(v_hat) + ε), μ_hat)
            
        else:
            X_inner -= np.multiply(η, grad)


        ### 4.2 Projection
        X_inner[X0==0.] = 0. # No new edges
        X_inner[X_inner<=0.] = 0. # No negative edges
        X_inner = normalize(X_inner, axis=1, norm='l1') # Row stochasticity

        
        ### 4.3 Objective value
        obj, _ = gradient_dense(np.asarray(X_inner), s.flatten(), obj_name=obj_name, is_directed=is_directed)
        feasible_objectives.append(obj)
        
        
        ### 4.4 Early stopping
        X_increment = scipy.linalg.norm(X_inner - X_prev)
        stats['increments'].append(X_increment)
        
        if t > 10 and early_stopping:
            Δ_obj = (np.min(feasible_objectives[-4:-3])/feasible_objectives[0] - np.min(feasible_objectives[-2:-1])/feasible_objectives[0])
            
            
            # A. Time limit
            if (time.time()-time0)/(60*60)>=5.:
                print(f"Time limit: stopping at t={t}, A increment", X_increment)
                stats['solve_time'] = time.time() - time0
                if is_directed:
                    return X_inner, feasible_objectives, stats
            
            
            
            # B. Tolerance
            if (np.sign(Δ_obj) == 1.) and Δ_obj < Δlimit:
                print(f"Convergence1: stopping at t={t},  Δ_obj%={Δ_obj:.5f} thr={Δlimit:.6f}, η={η:.5f}")
                stats['solve_time'] = time.time() - time0
                if is_directed:
                    return X_inner, feasible_objectives, None
                return np.diag(X_inner.sum(axis=0)) - X_inner, feasible_objectives, None # L= D - A --> A = D - L
            
            # C. Increment of loss
            elif (np.sign(Δ_obj) == -1.):
                print(f"Convergence2: stopping at t={t},  Δup={Δ_obj:.5f}")
                stats['solve_time'] = time.time() - time0
                if is_directed:
                    return X_inner, feasible_objectives, None 
                return X_inner, feasible_objectives, None
                               
        # D. Max iterations reached
        if t == max_iters and early_stopping:
            Δ_obj = ( np.mean(feasible_objectives[-4:-3])/feasible_objectives[0] - np.mean(feasible_objectives[-2:-1])/feasible_objectives[0] )
            print(f"early_stopping={early_stopping}, you reached {t}/{max_iters}, A increment {X_increment}, Δ_obj%={Δ_obj:.8f}, precision={X_inner.shape[0]*1e-5:.8f}. Please increase max_iters!")
        
    
    stats['solve_time'] = time.time() - time0
    print(f"    GD solution required {stats['solve_time']:.3f} sec")
    if is_directed:
        return X_inner, feasible_objectives, None
    return X_inner, feasible_objectives, None 