import argparse
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
sys.path += ['../src/', '../config/']
from data_loaders import load_real_dataset
from experiments_routines import do_experiment, select_nodes
from generative_graph_models import define_graph_instance
from generative_opinions_models import generate_opinions
from graph_signals import prepare_inputs, reconstruct_signal
from preprocessing import preprocess, define_initial_and_final_opinions, standardize_vec
from gnn import propagate_with_gcn
from label_propagation import propagate_with_label_prop
from random_reconstruction import random_reconstruction

from utils import set_mkl_threads; set_mkl_threads(10)
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="networkx")


save = True
parser = argparse.ArgumentParser(description="Eperiments to optimize opinions under opinion uncertainty.")
parser.add_argument('--exp', type=str, required=True, 
                    choices=["real-networks-directed", "real-networks-undirected", "synthetic-networks-directed", "synth-sensors", "real-sensors", "real-data", "network-size", "network-size2"])
parser.add_argument('--recmethod', type=str, required=True, choices=["gsignal", "gnn", "labelprop", "random"]) # labelprop IS THE MAIN REC METHOD FOR EXPERIMENTS
args = parser.parse_args()
type_of_experiment = args.exp
recmethod = args.recmethod


TIME0 = time.time()
SMOOTH_PERC = 0.15 # We assume signal is smooth (very optimistic choice for frequencies)
SENSORS_PERC = 0.20 # Larger than frequencies

if type_of_experiment == "real-networks-directed":
    is_directed = True
    is_synthetic = True
    NETWORKS = [('directed/out.moreno_highschool_highschool', {}),
                ('directed/out.dnc-temporalGraph', {}),
                ('directed/out.librec-ciaodvd-trust', {}),
                ('directed/out.librec-filmtrust-trust', {}),
                ('directed/out.moreno_health_health', {}),
                ('directed/out.moreno_innovation_innovation', {}),
                ('directed/out.moreno_oz_oz', {}),
                ('directed/out.wiki_talk_ht', {}),
                ]
    SENSORS_AND_FREQUENCIES = [(SENSORS_PERC, SMOOTH_PERC)]
    METHODS = ("random", "degree", "closeness_centrality", "pagerank") if recmethod == "labelprop" else ("degree",)
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[('gaussian', 3.), ('uniform', 1.)]
    NUMBER_OF_SEEDS = 50
    
elif type_of_experiment == "real-networks-undirected":
    is_directed = False
    is_synthetic = True
    NETWORKS = [('undirected/out.ucidata-zachary', {}), # 34
                ('undirected/out.moreno_beach_beach', {}), # 43
                ('undirected/out.moreno_train_train', {}), # 64
                ('undirected/out.mit', {}), # 96
                ('undirected/out.dimacs10-football', {}), # 115
                ]
    SENSORS_AND_FREQUENCIES = [(SENSORS_PERC, SMOOTH_PERC)]
    METHODS = ("random", "degree", "closeness_centrality", "pagerank")
    OBJECTIVES = ["PD"]
    OPINIONS_AND_POL=[('gaussian', 3.), ('uniform', 1.)]
    NUMBER_OF_SEEDS = 50

elif type_of_experiment == "synth-sensors":
    assert recmethod == "gsignal"
    is_synthetic = True
    is_directed = True
    n = 500
    NETWORKS = [("erdos", {'n': n, 'p': 0.25}),
    ]
    # Define params 
    _FREQ = int(n*SMOOTH_PERC) # numb frequencies as % of nodes
    distance = 50 # the distance in the grid 
    max_value = 475 # the maximum value in the grid

    # grid definition
    SENSORS_AND_FREQUENCIES = []
    for i in range(_FREQ, max_value + distance, distance):
        for j in range(_FREQ, i + distance, distance):  # Allow j to be equal to i
            SENSORS_AND_FREQUENCIES.append((i, j))
    METHODS = ["random",]
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[('gaussian', 3.), ('uniform', 1.)]
    NUMBER_OF_SEEDS = 50

elif type_of_experiment == "real-sensors":
    is_synthetic = False
    is_directed = True
    NETWORKS = [("referendum", {})] # Referendum: 2479 

    # grid definition
    SENSORS = np.linspace(25, 2500, 10)
    SENSORS[-1] = 2479
    SENSORS_AND_FREQUENCIES = [(int(x), int(x)-10) for x in SENSORS]
    METHODS = ["degree",]
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[(None, None)]
    NUMBER_OF_SEEDS = 1

elif type_of_experiment == "network-size":
    is_synthetic = True
    is_directed = True
    NETWORKS = erdos_params = [
        ("erdos", {'n': 100, 'p': 0.25}),
        ("erdos", {'n': 300, 'p': 0.25}),
        ("erdos", {'n': 500, 'p': 0.25}),
        ("erdos", {'n': 700, 'p': 0.25}),
        ("erdos", {'n': 900, 'p': 0.25}),
        ("erdos", {'n': 1_100, 'p': 0.25}),
        ("erdos", {'n': 1_300, 'p': 0.25}),
        ("erdos", {'n': 1_500, 'p': 0.25}),
        ("erdos", {'n': 1_700, 'p': 0.25}),
        ("erdos", {'n': 1_900, 'p': 0.25}),
        ("erdos", {'n': 2_100, 'p': 0.25}),
        ("erdos", {'n': 2_300, 'p': 0.25}),
        ("erdos", {'n': 2_500, 'p': 0.25}),
        ("erdos", {'n': 2_700, 'p': 0.25}),
        ("erdos", {'n': 2_900, 'p': 0.25}),
        ("erdos", {'n': 3_100, 'p': 0.25}),
        ("erdos", {'n': 3_300, 'p': 0.25}),
        ("erdos", {'n': 3_500, 'p': 0.25}),
        ("erdos", {'n': 3_700, 'p': 0.25}),
        ("erdos", {'n': 3_900, 'p': 0.25}),
        ("erdos", {'n': 4_100, 'p': 0.25}),
        ("erdos", {'n': 4_300, 'p': 0.25}),
        ("erdos", {'n': 4_500, 'p': 0.25}),
        ("erdos", {'n': 4_700, 'p': 0.25}),
        ("erdos", {'n': 4_900, 'p': 0.25}),
        ("erdos", {'n': 5_000, 'p': 0.25})
    ]
    SENSORS_AND_FREQUENCIES = [(SENSORS_PERC, SMOOTH_PERC)]
    METHODS = ["random",]
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[('uniform', 3.)]
    NUMBER_OF_SEEDS = 25

elif type_of_experiment == "network-size2":
    is_synthetic = True
    is_directed = True
    m = 5
    NETWORKS = erdos_params = [
        ("barabasi", {'n': 100, 'm': m}),
        ("barabasi", {'n': 300, 'm': m}),
        ("barabasi", {'n': 500, 'm': m}),
        ("barabasi", {'n': 700, 'm': m}),
        ("barabasi", {'n': 900, 'm': m}),
        ("barabasi", {'n': 1_100, 'm': m}),
        ("barabasi", {'n': 1_300, 'm': m}),
        ("barabasi", {'n': 1_500, 'm': m}),
        ("barabasi", {'n': 1_700, 'm': m}),
        ("barabasi", {'n': 1_900, 'm': m}),
        ("barabasi", {'n': 2_100, 'm': m}),
        ("barabasi", {'n': 2_300, 'm': m}),
        ("barabasi", {'n': 2_500, 'm': m}),
        ("barabasi", {'n': 2_700, 'm': m}),
        ("barabasi", {'n': 2_900, 'm': m}),
        ("barabasi", {'n': 3_100, 'm': m}),
        ("barabasi", {'n': 3_300, 'm': m}),
        ("barabasi", {'n': 3_500, 'm': m}),
        ("barabasi", {'n': 3_700, 'm': m}),
        ("barabasi", {'n': 3_900, 'm': m}),
        ("barabasi", {'n': 4_100, 'm': m}),
        ("barabasi", {'n': 4_300, 'm': m}),
        ("barabasi", {'n': 4_500, 'm': m}),
        ("barabasi", {'n': 4_700, 'm': m}),
        ("barabasi", {'n': 4_900, 'm': m}),
        ("barabasi", {'n': 5_000, 'm': m})
    ]
    SENSORS_AND_FREQUENCIES = [(SENSORS_PERC, SMOOTH_PERC)]
    METHODS = ["random",]
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[('gaussian', 3.)]
    NUMBER_OF_SEEDS = 50

elif type_of_experiment == "real-data":
    is_synthetic = False
    is_directed = True
    n = None
    NETWORKS = [("brexit", {}), ("referendum", {}), ("vaxNoVax", {})] 
    SENSORS_AND_FREQUENCIES = [(SENSORS_PERC, SMOOTH_PERC)]
    METHODS = ("random", "degree", "closeness_centrality", "pagerank")
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[(None, None)]
    NUMBER_OF_SEEDS = 1

elif type_of_experiment == "synthetic-networks-directed":
    is_synthetic = True
    is_directed = True
    n = 500
    NETWORKS = [("sbm", {'n': n}),
                ("barabasi", {'n': n, 'm': 4}),
                ("erdos", {'n': n, 'p': 0.25}),
    ]
    SENSORS_AND_FREQUENCIES = [(SENSORS_PERC, SMOOTH_PERC)]
    METHODS = ("random", "degree", "closeness_centrality", "pagerank")
    OBJECTIVES = ["PD", "D", "P"]
    OPINIONS_AND_POL=[('gaussian', 3.), ('uniform', 1.)]
    NUMBER_OF_SEEDS = 50

else:
    raise Exception(f"type_of_experiment={type_of_experiment} not defined")


params_combination = len(NETWORKS) * len(SENSORS_AND_FREQUENCIES) * len(OPINIONS_AND_POL) * len(OBJECTIVES) * NUMBER_OF_SEEDS * len(METHODS)



combination_index = 1
for network_name, kwargs in NETWORKS:
    result = {'method': [],
              'bound': [], 
              'optimization_error': [], 'reconstr_error': [],
              'network_name': [],  'network_nodes': [], 'network_edges': [],
              'n_frequencies': [], 'n_sensors': [],
              'is_synthetic': [], 'is_directed': [], 
              'opinion_model': [], 'polarization': [], 
              'o_seed': [], 
              'obj_name': [],
              'obj_with_reconstr': [], 'best_obj': [], 'relative_optimization_error': [], 'max_λ': [],
              'time_sec': [], 
             }


    # 1. Load -------------------------
    seeds = list(range(NUMBER_OF_SEEDS)) if is_synthetic else [0,]
    A_eq, G, _L_eq = define_graph_instance(network_name, kwargs=kwargs, directed=is_directed)
    if isinstance(G, nx.MultiDiGraph):
        print(f"Skipping Multigraph {network_name}")
        nx.write_graphml(G, f"./{network_name}-multigraph.graphml")
        continue

    # 2. File name and check existence
    __network_name = network_name.split("/")[1] if "/" in network_name else network_name
    output_path = Path(f"../data/processed-{recmethod}/")
    file_path = output_path / f"results-{type_of_experiment}-{__network_name}-nodes{G.number_of_nodes()}.csv"
    if file_path.exists():
        print(f"{file_path} exists, skipping.")
        continue
    
    # 3. Loop parameters
    for n_sensors, n_frequencies in SENSORS_AND_FREQUENCIES:
        if type_of_experiment not in ("synth-sensors", "real-sensors"): # if testing sensors, "SENSORS_AND_FREQUENCIES contains exact numbers (int)
            n_sensors = int(G.number_of_nodes() * n_sensors)
            if recmethod == "gsignal":
                n_frequencies = int(G.number_of_nodes() * n_frequencies)
        for opinion_model, pol in OPINIONS_AND_POL:
            for obj_name in OBJECTIVES:
                for o_seed in seeds:
                    print("\n\n\n\n")
                    print("===="*25)
                    print('obj_name', obj_name, 'o_seed', o_seed, 'pol', pol, 'opinion_model', opinion_model, 
                            'network_name', network_name, 'is_directed', is_directed)
                    if type_of_experiment not in ("real-data", "real-sensors"): # in these two experiments we use real opinions
                        x = generate_opinions(o_name=opinion_model, G=G, pol=pol, seed=o_seed)
                    else:
                        x, _G = load_real_dataset(network_name, verbose=False)
                    
                    x, _, I = preprocess(x, G)
                    s, _z_eq = define_initial_and_final_opinions(x=x, M_eq = I + np.diag(A_eq.sum(axis=1)) - A_eq)

                    print(f"Graph has {len(s)} nodes, {len(A_eq.data)} variables, {np.quantile(np.array(A_eq.sum(axis=1)).flatten(), .5):.3f} avg degree ")
                    print("Loading files  ✅")


                    # 2. Best  -------------------------
                    print("Compute best objective value  ..")
                    best_obj, _ = do_experiment(A_eq, _L_eq, s, s, obj_name=obj_name, is_directed=is_directed)
                    print("Compute best objective value  ✅")


                    # 3. Initialize signal processing -----------
                    print("Preprocess graph signals ..")
                    n = G.number_of_nodes()
                    
                    if recmethod == "gsignal":
                        classes = np.asarray(s > np.mean(s)).flatten().astype(int)
                        U, U_f, R, L, λ = prepare_inputs(G, n_frequencies, classes, 0., verbose=True)

                        S = np.zeros(n)
                        R = np.diag(np.repeat(0.05, n))
                        R_inv = np.linalg.inv(R)
                    print("Preprocess signals  ✅")

                    
                    print("----"*25)
                    print("\n\n")


                    # 4. Grid
                    for method in METHODS:
                        assert len(method) > 1, f"Method error: {method}"
                        time0 = time.time()
                        # --------------------------
                        # 4.1 Sensors selection
                        selected_nodes = select_nodes(G, n_sensors, method)
                        if recmethod == "gsignal":
                            S[selected_nodes] = 1
                            D_s = np.asmatrix(np.zeros((n, n)))
                            np.fill_diagonal(D_s, S)
                            P_s = D_s[:, selected_nodes]

                        print("sensor selection ✅")


                        # ---------------------------
                        # 4.2 Reconstructing signal
                        if recmethod == "gsignal":
                            s_rec, sensors, max_λ = reconstruct_signal(s, U, U_f, 0, n_sensors, R, normalize=False, verbose=True, selected_nodes=selected_nodes)
                        elif recmethod == "gnn":
                            s_rec, sensors, max_λ = propagate_with_gcn(G, selected_nodes, s)
                        elif recmethod == "labelprop":
                            s_rec, sensors, max_λ = propagate_with_label_prop(G, selected_nodes, s)
                        elif recmethod == "random":
                            s_rec, sensors, max_λ = random_reconstruction(G, selected_nodes, s)
                        s_cast = standardize_vec(np.clip(s_rec, np.min(s), np.max(s)))
                        print("signal reconstruction ✅")

                        reconstr_error = np.sqrt(np.sum((s_cast.flatten() - s.flatten())**2))
                        bound =  0. # to be filled in dataframe after
                        
                        _, obj_with_reconstr = do_experiment(A_eq, _L_eq, s_cast, s, obj_name=obj_name, is_directed=is_directed)
                        optimization_error = np.sqrt(np.sum((obj_with_reconstr - best_obj)**2)) # RMSE
                        relative_optimization_error = obj_with_reconstr / best_obj
                        
                        print("-"*10)
                        print(f"\n\n Method: {method},  Recmethod: {recmethod}")
                        print(f"reconstr_error: {reconstr_error:.3f}  bound={bound:.3f}", )
                        print(f"optimization_error: {optimization_error:.3f}", f"bound: {bound:.3f}    ", f" : {obj_with_reconstr:.2f} vs {best_obj:.2f} \n\n")
                        print(f"{combination_index}/{params_combination}  time={time.time()-time0:.3f}")
                        print("-"*10)
                        print("\n")
                        combination_index += 1
                        
                        
                        result['method'].append(method)
                        result['reconstr_error'].append(reconstr_error)
                        result['obj_with_reconstr'].append(obj_with_reconstr)
                        result['best_obj'].append(best_obj)
                        result['relative_optimization_error'].append(relative_optimization_error)
                        result['optimization_error'].append(optimization_error)
                        result['bound'].append(bound)
                        result['network_name'].append(network_name)
                        result['network_nodes'].append(G.number_of_nodes())
                        result['network_edges'].append(G.number_of_edges())
                        result['n_frequencies'].append(n_frequencies)
                        result['n_sensors'].append(n_sensors)
                        result['max_λ'].append(max_λ)
                        result['is_synthetic'].append(is_synthetic)
                        result['is_directed'].append(is_directed)
                        result['opinion_model'].append(opinion_model)
                        result['polarization'].append(pol)
                        result['o_seed'].append(o_seed)
                        result['obj_name'].append(obj_name)
                        result['time_sec'].append(time.time()-time0)
        


        df = pd.DataFrame(result)
    if save:
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

print(f"\n\n FINAL TIME (min): {(time.time()-TIME0)/60:.3f}")