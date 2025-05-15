import numpy as np
import networkx as nx


def sampling_sensors(n:int, U: np.matrix, strategy, R_inv: np.matrix) -> set:
    '''Returns set of n sensors 
    '''
    # trace
    S = set()
    nodes = set(range(U.shape[0]))
    D = np.zeros_like(U)
    if strategy == 0:
        return np.random.choice(list(range(U.shape[0])), n, replace=False)
    elif strategy==1:
        target_function = lambda D_curr: (None, np.trace(np.linalg.pinv(U.T@D_curr@R_inv@U)))
    elif strategy==2:
        target_function = lambda D_curr: np.linalg.slogdet(U.T@D_curr@R_inv@U)
    else:
        print('Not implemented')
    
    while len(S) < n:
        print(f"sensors: {len(S)}/{n}", end="\r")
        max_target_value = -np.inf
        idx_max_target_value = None
        for i, node in enumerate(nodes - S):
            D_curr = D.copy()
            D_curr[node, node] = 1
            _sign, target_value_curr = target_function(D_curr)
            print(f"{i+1}/{len(nodes - S)}", target_value_curr)
            if max_target_value < target_value_curr:
                max_target_value = target_value_curr
                idx_max_target_value = node 
        #print(idx_max_target_value, max_target_value)
        S = S.union(set([idx_max_target_value]))
        D[idx_max_target_value, idx_max_target_value] = 1
    return S


def prepare_inputs(G: nx.DiGraph, k: int, classes: np.array, 
                   cross_corr:float, verbose:bool=False) -> (np.matrix, np.matrix):
    '''Returns the Eigenvector matrix of the laplacian 
       and the reduced version
       - G := graph instance 
       - x := graph signal x ∈ R^|V|
       - k := number of selected frequencies (selected in decreasing order)
    '''
    G = G.to_undirected()
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1).reshape(-1).tolist()[0])
    I = np.diag(np.ones(A.shape[0]))
    L = D - A
    assert (L - nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()).sum() == 0

    λ, U = np.linalg.eigh(L)
    if verbose:
        print('Eigendecomposition error: ', (L - U @ np.asmatrix(np.diag(λ)) @ U.T).sum())
    
    #choosing frequencies with highest eigenvalue
    frequencies_sorted = sorted(list(range(len(λ))), key=lambda i: np.absolute(λ[i]), reverse=False)
    frequencies_selected = frequencies_sorted[:k]
    
    # GFT
    U_f = U[:, frequencies_selected]
    
    # covariance matrix for signals with noise
    R = prepare_covariance_mtx(classes, cross_corr)
    
    if verbose:
        return U, U_f, R, L, λ
    
    return U, U_f, R

def prepare_covariance_mtx(classes, cross_corr:float) -> np.matrix:
    R = np.diag(np.zeros(len(classes)))
    for i in range(len(classes)):
        R[i, (classes == classes[i])] = + cross_corr
        R[i, (classes != classes[i])] = - cross_corr
    np.fill_diagonal(R, 1.)
    return R


def precompute_inputs(U: np.matrix, U_f: np.matrix, numb_sensors: int, strategy: int, R: np.matrix=None, selected_nodes: np.array=None):
    '''Returns the Q matrix and the selected sampled nodes using given strategy
    '''
    n = U.shape[0]
    
    S = np.zeros(n)
    R_inv = np.linalg.inv(R)
    if selected_nodes is None:
        selected_nodes = np.array(sorted(sampling_sensors(numb_sensors, U, strategy, R_inv)))#np.random.choice(list(range(n)), 10)
    S[selected_nodes] = 1
    
    D_s = np.asmatrix(np.zeros((n, n)))
    np.fill_diagonal(D_s, S)

    P_s = D_s[:, selected_nodes]
    I = np.diag(np.ones(n))
    
    M = np.linalg.inv(P_s.T @ R @ P_s)
    Q = np.linalg.inv(U_f.T @ P_s @ M @ P_s.T @ U_f) @ U_f.T @ P_s @ M
    #Q = np.linalg.inv(U_f.T @ D_s @ U_f) @ U_f.T @ P_s #np.linalg.pinv(P_s.T @ U_f)

    Ds_bar = I - D_s
    s_test = np.linalg.svd(np.dot(Ds_bar, U_f), compute_uv=False)
    phrase = 'BAD: Spectral norm (Max singular value) not less than 1! ' + str(max(s_test)) if max(s_test)>= 1 else 'OK: Spectral norm (Max singular value) less than 1! ' + str(max(s_test))
    #assert max(s_test) < 1, phrase
    print(phrase)

    
    return Q, selected_nodes, max(s_test)


def reconstruct_signal(x, U, U_f, strategy, numb_sensors, R, normalize=True, verbose=False, selected_nodes=None):
    '''no noise   U_f [  @(U_f.T      @ D         @U_f )^-1 @ U_f.T @ P_s    ]  @ x_s
       with noise U_f [  @(U_f.T @P_s @ M @ P_s.T @U_f )^−1 @ U_f.T @ P_s @M ]  @ x_s
                  M = (P_s.T @ R @ P_s)^−1.
    '''
    print("Pre-computing matrices .. ")
    Q, selected_nodes, max_λ = precompute_inputs(U, U_f, numb_sensors, strategy, R, selected_nodes)
    
    x_s = np.asmatrix(x[selected_nodes]).reshape(len(selected_nodes), 1)

    print("Pre-computing matrices  ✅")

    print("Reconstructing signal .. ")
    x_reconstructed = np.asarray((U_f @ Q @ x_s).reshape(-1))[0]
    if normalize:
        x_reconstructed = (x_reconstructed - np.min(x_reconstructed)) / (np.max(x_reconstructed) - np.min(x_reconstructed))
    if verbose:
        print('Squared error', np.sum((x - x_reconstructed)**2) / np.sum(x**2))
        return x_reconstructed, selected_nodes, max_λ
    return x_reconstructed, selected_nodes, max_λ
