import ctypes
import networkx as nx
import numpy as np
import random
import scipy
from sklearn.preprocessing import normalize

def set_seed(seed_value):
    # Set the seed for numpy
    np.random.seed(seed_value)
    
    # Set the seed for Python's built-in random module (used by networkx)
    random.seed(seed_value)


#####################################################################       
###########################    PROJECTION    ########################
#####################################################################       
def check_graph(s: np.array, G: nx.Graph):
    """Returns weakly connected graph with correspondent opinions
    """
    # 1. Setting opinions
    opinions = {node: opinion for node, opinion in zip(G.nodes(), np.asarray(s).flatten())}
    nx.set_node_attributes(G, opinions, "opinion")

    # 2. Removing nodes
    directed = G.is_directed()
    if (directed) and (not nx.is_weakly_connected(G)):
        print("   ! Selecting largest weakly connected component and relabeling nodes..")
        Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)[0]
        G = G.subgraph(Gcc)
        G = nx.convert_node_labels_to_integers(G)
            
    if directed:
        remove = [node for node, degree in dict(G.out_degree()).items() if degree == 0]
        i = 0
        while len(remove)>0:
            print(f"   ! Removing out-disconnected nodes.. \n  iter={i}, remaining {G.number_of_nodes()}")
            nodes = list(G.nodes())
            for u in remove:
                nodes.remove(u)
            G = G.subgraph(nodes)
            G = nx.convert_node_labels_to_integers(G)
            remove = [node for node, degree in dict(G.out_degree()).items() if degree == 0]
            i += 1

        # Assert there are no self-loops in the subgraph
        assert not any(G.has_edge(n, n) for n in G.nodes), "The subgraph contains self-loops."
    
    # 3. Returning new graph and opinions
    return np.array([data['opinion'] for _, data in G.nodes(data=True)]).reshape((G.number_of_nodes(), 1)), G



def cast_to_doubly_stochastic(M: scipy.sparse.csr, T: int = 10_000, 
                              ε: float = 1e-3, verbose : bool = True):
    for i in range(T):
        M = normalize(M, axis=1, norm='l1')
        M = normalize(M, axis=0, norm='l1')
        if verbose:
            print(f"   casting adj to doubly stochastic {i}/{T}", end="\r")
        if np.all(np.absolute(M.sum(axis=1) - 1) < ε):
            if verbose:
                print(f"   Finished casting adj to doubly stochastic at iteration {i}/{T}")
            return M
    assert np.all(M.todense()>=0.)
    print("Increase T")
    
    
    
#####################################################################       
###########################    TESTING    ###########################
#####################################################################

def is_symmetric(a, tol=1e-6):
    return np.all(np.abs(a - a.T) < tol)

def check_results(L_opt, s, A=None, b=None, ε=0.001):
    I = np.identity(L_opt.shape[0])
    __full_print = True
    if b is None:
        b = np.diagonal(L_opt)
        __full_print = False
    check_laplacian, check_budget = (np.all(np.absolute(L_opt.sum(axis=1))<ε), np.all((np.diagonal(L_opt)-b)<=ε))
    text = f'Symmetry: {is_symmetric(L_opt)},  Is laplacian: {check_laplacian}'
    if not check_laplacian:
        print(f"max/min sum row: {np.max(np.absolute(L_opt.sum(axis=1))):.3f}/{np.min(np.absolute(L_opt.sum(axis=1))):.3f}")
    text += f', Budget: {check_budget}' if __full_print else '' 
    print(text)
    print(f"L has negative entries: {np.all([L_opt[i, j] <= 0 for i in range(len(s)) for j in range(len(s)) if i!=j])}")
    if A is not None:
        print(f'No new edges: {np.all([L_opt[i, j] <= ε for i, j in np.argwhere((A + I) == 0)])}')
    print(f'objective value is: {(s.T@np.linalg.inv(np.identity(L_opt.shape[0])+ L_opt)@s).item():.3f}')
    print(f'Trace is: {np.trace(L_opt):.3f}')

    
#####################################################################       
###########################    MULTIPROCESSING    ###################
#####################################################################


def set_mkl_threads(n):
    """Sets all the env variables to limit numb of threads
    """
    try:
        import mkl
        mkl.set_num_threads(n)
        return 0
    except:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:  # pylint: disable=bare-except
            pass
    v = f"{n}"
    os.environ["OMP_NUM_THREADS"] = v  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = v  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = v  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = v  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = v  # export NUMEXPR_NUM_THREADS=6