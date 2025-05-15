import networkx as nx
import numpy as np
import scipy
from time import time
import sys
sys.path += ['../src/', '../config/']
from algorithms_approx import GD_optimizer, gradient_dense
from algorithms_optimal import cvx_optimizer
from generative_graph_models import assert_adj_stochasticity


def select_nodes(G, n_sensors, method):
    if method == "degree":
        centralities = dict(G.degree())
    elif method == "in_degree":
        centralities = nx.in_degree_centrality(G)
    elif method == "out_degree":
        centralities = nx.out_degree_centrality(G)
    elif method == "eigenvector_centrality":
        centralities = nx.eigenvector_centrality(G, max_iter=100)
    elif method == "closeness_centrality":
        centralities = nx.closeness_centrality(G)
    elif method == "pagerank":
        centralities = nx.pagerank(G)
    elif method == "hits":
        hubs, authorities = nx.hits(G)
        centralities = hubs
    elif method == "random":
        return np.random.choice(list(range(G.number_of_nodes())), n_sensors, replace=False)
    
    else:
        raise Exception(f"{method} not implemented")

    if not isinstance(centralities, list):
        centralities = list(centralities.items())
    top_nodes = sorted(centralities, key=lambda x: x[1], reverse=True)[:n_sensors]
    return  np.array([node for node, _ in top_nodes])

def do_experiment(A_eq, _L_eq, s, s_true, obj_name, is_directed):
    """returns f^, f* at equilibrium
       f^ depends on s^, f* depends on s_true
    """
    η = 0.2
    max_iters = 100
    early_stopping = True
    routine = 'ADAM'
    grad_params = {}
    lr_params = {"lr_coeff": η}
    verbosity = 1
    ε = 1e-6
    ε_cvx = 1e-2
    max_iters_cvx = 500


    # 0. Optimization routine:
    time0 = time()
    if is_directed:
        X, objectives, _stats = GD_optimizer(A_eq, s, max_iters,
                                            early_stopping=early_stopping,
                                            grad_params=grad_params,
                                            lr_params=lr_params,
                                            verbosity=verbosity,
                                            obj_name=obj_name,
                                            is_directed=is_directed,
                                            ε=ε,
                                            routine=routine,
        )  # X is the optimized adj matrix
    else:
        X, objectives, _stats = cvx_optimizer(A_eq, s, eps=ε_cvx, max_iters=max_iters_cvx) # X is the optimized adj matrix
    
    Δt = time() - time0
    
    # 1. First objective
    initial_index, _ = gradient_dense(np.asarray(A_eq), s.flatten(), obj_name=obj_name, is_directed=is_directed)

    # 2. Natural objective (using input graph)
    obj_nat, _ = gradient_dense(np.asarray(A_eq), s.flatten(), obj_name=obj_name, is_directed=is_directed)

    # 3. Optimized objective: with A^ and s^
    obj, _ = gradient_dense(np.asarray(X), s.flatten(), obj_name=obj_name, is_directed=is_directed)
    objectives.append(obj)

    #print((f"   --> Method  obtained obj%={(1-obj/obj_nat)*100:.4f}% ", f"  total reduc={(1-obj/initial_index)*100:.4f}% in {Δt:.3f}sec"))

    # 4. Finding true objective value:
    true_obj, _ = gradient_dense(np.asarray(X), s_true.flatten(), obj_name=obj_name, is_directed=is_directed)
    
    #print(f"\nlast objective: {objectives[-1]:.3f}    natural objective: {obj_nat:.3f}    true objective: {true_obj:.3f}")
    return objectives[-1], true_obj