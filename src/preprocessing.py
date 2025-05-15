import networkx as nx
import numpy as np
import scipy
from sklearn.preprocessing import normalize

standardize_vec = lambda x, scale=1: scale*(x - np.mean(x)) / np.std(x)
normalize_vec = lambda x, lb, up: (up - lb) * (x - np.min(x)) / (np.max(x) - np.min(x)) + lb

def preprocess(x,
               G,
               standardize_x: bool = True,
               normalize_x: bool = False,
               normalize_A: bool = False,
               o_min: float = -.5, o_max: float = .5,
    ):
    assert int(standardize_x) + int(normalize_x) == 1, "Choose among standardization or normalization"
    if standardize_x:
        print("Standardize opinion vec")
        x = standardize_vec(x)
    elif normalize_x:
        print("Normalizing opinion vec")
        x = normalize_vec(x, o_min, o_max)
    if normalize_A:
        print("Normalizing adjacency mtx")
        A_eq = normalize(nx.adjacency_matrix(G), axis=1, norm='l1')
        return x.reshape((len(x), 1)), A_eq.toarray().astype(np.float64), scipy.sparse.identity(A_eq.shape[0])

    return x.reshape((len(x), 1)), nx.adjacency_matrix(G).toarray().astype(np.float64), scipy.sparse.identity(len(x))


def define_initial_and_final_opinions(x, M_eq):
    """Returns the matched pair (z_eq, s) according to FJ model:
        directed: M_eq = 2*I - A_eq = I + L
        undirected: M_eq = (I + D ) - A_eq = I + L
    """
    print("Computing the pair of inner opinions and equilibirum opinions ..")
    s_int = standardize_vec(np.array(M_eq @ x), scale=1)
    z_eq = x

    return s_int, z_eq