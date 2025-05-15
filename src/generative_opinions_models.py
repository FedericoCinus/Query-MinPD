import numpy as np
import networkx as nx
import sys
sys.path.append('../src')
from utils import set_seed

polarization_f = lambda x, pol: np.power(np.absolute(x), 1 / pol) if x >= 0 else -np.power(np.absolute(x), 1 / pol)
polarization_f = np.vectorize(polarization_f)


def generate_opinions(o_name: str,
                      G: nx.Graph, 
                      pol: float = 1.,
                      standardize_x: bool = True,
                      normalize_x: bool = False,
                      o_min: float = -.5, o_max: float = .5,
                      seed: int = None
    ):
    assert int(standardize_x) + int(normalize_x) == 1, "Choose among standardization or normalization"

    if seed is not None:
        set_seed(seed)
        np.random.seed(seed)

    n = G.number_of_nodes()
    if o_name in {'uniform', 'constant'}:
        if o_name == 'uniform':
            opinions = np.random.uniform(o_min, o_max, size=(n, 1))
        elif o_name == 'constant':
            assert o_max == o_min, f'Set o_max ({o_max}) = o_min ({o_min})'
            opinions = np.repeat(o_min, n).reshape((n, 1))
            normalized = None
            print(f'Constant opinions with polarization {pol}, normalized = {normalized}')
            
        opinions = polarization_f(opinions, pol) # insert polarization
    
    # community-based opinions
    elif o_name == 'gaussian':
        communities = nx.algorithms.community.kernighan_lin_bisection(G.to_undirected())
        opinions = sampling_gaussian_opinions(n, communities, pol, o_min, o_max, seed)

    return opinions.reshape((len(opinions), 1))


def sampling_gaussian_opinions(n, communities, pol, o_min, o_max, seed):
    if seed:
        set_seed(seed)
        np.random.seed(seed)
    
    # computing the range and centre of the opinion spectrum in input
    Δopinion, middle_point = ((o_max - o_min)/2, (o_max + o_min)/2)
    δopinion = Δopinion / ((len(communities))//2)
    
    # defining the locations of the normal distributions on for each community
    μs= np.zeros(len(communities)) 
    
    # computing pairs of opposite opinion locations with distance proportional to pol
    for i in range(0, len(communities)//2+1, 2):
        μs[i] = middle_point + (.1*pol) * δopinion * (i+1)
        μs[i+1] = middle_point - (.1*pol) * δopinion * (i+1)
    
    # sampling community opinions
    opinions = np.zeros(n)
    for community_idx, users in enumerate(communities):
        for u in users:
            opinions[u] = np.random.normal(μs[community_idx], Δopinion/6)
    return opinions
