import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import sys
sys.path += ['../src/', '../config/']
from utils import check_graph


def load_undirected_graph(filepath):
    # Create an empty undirected graph
    G = nx.Graph()

    # Read the edgelist from the file and store edges temporarily
    edges = []
    min_weight = float('inf')

    with open(filepath, 'r') as file:
        for line in file:
            # Skip lines that start with "%"
            if line.startswith("%"):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                source = parts[0]
                target = parts[1]

                # Check if there is a third column for weight
                if len(parts) >= 3:
                    try:
                        weight = float(parts[2])
                    except ValueError:
                        continue
                else:
                    weight = 1.0  # Default weight if not provided

                # Skip self-loops
                if source == target:
                    continue

                edges.append((source, target, weight))
                if weight < min_weight:
                    min_weight = weight

    # Adjust weights to be positive if there are negative weights
    if min_weight < 0:
        shift = abs(min_weight)
        edges = [(source, target, weight + shift) for source, target, weight in edges]

    # Add edges to the graph
    for source, target, weight in edges:
        if not G.has_edge(source, target):
            G.add_edge(source, target, weight=weight)
        else:
            G[source][target]['weight'] += weight

    # Assert the graph is undirected
    assert not nx.is_directed(G), "The graph is directed."

    # Get the largest connected component
    largest_connected_component = max(nx.connected_components(G), key=len)
    
    # Create a subgraph with only the largest connected component
    G_sub = G.subgraph(largest_connected_component).copy()

    # Assert there are no self-loops in the subgraph
    assert not any(G_sub.has_edge(n, n) for n in G_sub.nodes), "The subgraph contains self-loops."

    # Assert there are no multiedges (Graph does not support multiedges)
    assert all(G_sub.number_of_edges(u, v) == 1 for u, v in G_sub.edges), "The subgraph contains multiedges."

    # Assert the subgraph is the largest connected component
    assert nx.is_connected(G_sub), "The subgraph is not connected."

    # Relabel nodes to integers from 0 to n-1
    mapping = {node: i for i, node in enumerate(G_sub.nodes())}
    G_sub = nx.relabel_nodes(G_sub, mapping)

    return G_sub

def load_directed_graph(filepath):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Read the edgelist from the file and store edges temporarily
    edges = []
    min_weight = float('inf')

    with open(filepath, 'r') as file:
        for line in file:
            # Skip lines that start with "%"
            if line.startswith("%"):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                source = parts[0]
                target = parts[1]

                # Check if there is a third column for weight
                if len(parts) >= 3:
                    try:
                        weight = float(parts[2])
                    except ValueError:
                        continue
                else:
                    weight = 1.0  # Default weight if not provided

                # Skip self-loops
                if source == target:
                    continue

                edges.append((source, target, weight))
                if weight < min_weight:
                    min_weight = weight

    # Adjust weights to be positive if there are negative weights
    if min_weight < 0:
        shift = abs(min_weight)
        edges = [(source, target, weight + shift) for source, target, weight in edges]

    # Add edges to the graph
    for source, target, weight in edges:
        if not G.has_edge(source, target):
            G.add_edge(source, target, weight=weight)
        else:
            G[source][target]['weight'] += weight

    # Assert the graph is directed
    assert nx.is_directed(G), "The graph is not directed."

    # Get the largest weakly connected component
    largest_weakly_connected_component = max(nx.weakly_connected_components(G), key=len)
    
    # Create a subgraph with only the largest weakly connected component
    G_sub = G.subgraph(largest_weakly_connected_component).copy()

    # Assert there are no self-loops in the subgraph
    assert not any(G_sub.has_edge(n, n) for n in G_sub.nodes), "The subgraph contains self-loops."

    # Assert there are no multiedges (DiGraph does not support multiedges)
    assert all(G_sub.number_of_edges(u, v) == 1 for u, v in G_sub.edges), "The subgraph contains multiedges."

    # Assert the subgraph is the largest weakly connected component
    assert nx.is_weakly_connected(G_sub), "The subgraph is not weakly connected."

    # Relabel nodes to integers from 0 to n-1
    mapping = {node: i for i, node in enumerate(G_sub.nodes())}
    G_sub = nx.relabel_nodes(G_sub, mapping)

    return G_sub


def load_real_dataset(name: str, use_follow_net: bool=True, verbose: bool=True) -> (np.array, nx.DiGraph):
    """Returns opinion array, and DiGraph
    """

    if name in ("brexit", "vaxNoVax", "referendum"):
        g_folder = Path(f"../data/raw/{name}/edgelist.txt")
        o_folder = Path(f"../data/raw/{name}/propagations_and_polarities.pkl")
        with open(o_folder, "rb") as f:
            propagations, polarities = pickle.load(f)

        # Our convention is: u --> v, as in FJ model
        if use_follow_net: 
            # u follows v: so v can influence u
            if verbose:
                print(f"Loading {name} follow network")
            G = nx.read_edgelist(g_folder, create_using=nx.DiGraph)
        else: 
            # u --> v: v is exposed to u
            if verbose:
                print(f"Loading {name} exposure retweet network") 
            G = build_retweet_exposure_graph(propagations)
            G = nx.DiGraph.reverse(G) # Because the semantic is inverted


        x = compute_average_opinion(propagations, polarities, G)
        

    x, G = check_graph(x, G)
    if verbose:
        print(f"Graph is directed {G.is_directed()},  |V|={G.number_of_nodes():_},  |E|={G.number_of_edges():_}")



    return x, G


def load_PC(axes, weighted: bool = False):

    comments_df = pd.read_csv(dataset_folder_PC / "submissions_anonymized_PC.csv")
    edges_df = pd.read_csv(dataset_folder_PC / "edges_anonymized_PC.csv")
    
    #  Graph weights
    edges_df['sentiment'] = edges_df['sentiment'] - edges_df['sentiment'].min() + 1 if weighted else 1
    G = nx.from_pandas_edgelist(edges_df, 'parent', 'child', ['sentiment'], create_using=nx.DiGraph())

    # Assign scalar values based on author_flair_text
    if axes == "LR":
        labels = {':libleft: - LibLeft', ':libright: - LibRight',
                ':authleft: - AuthLeft', ':right: - Right', ':lib: - LibCenter',
                ':auth: - AuthCenter', ':left: - Left', ':authright: - AuthRight',
                ':centrist: - Centrist'}
        # Drop na
        comments_df = comments_df[comments_df.author_flair_text.isin(labels)]

        comments_df.loc[comments_df['author_flair_text'].str.contains('Right'), 'opinion'] = 1
        comments_df.loc[comments_df['author_flair_text'].str.contains('Left'), 'opinion'] = -1
        comments_df.loc[comments_df['author_flair_text'].str.contains('Center'), 'opinion'] = 0
        comments_df.loc[comments_df['author_flair_text'].str.contains('Centrist'), 'opinion'] = 0
    
    elif axes == "AL":
        labels = {':libleft: - LibLeft', ':libright: - LibRight',
                ':authleft: - AuthLeft',  ':lib: - LibCenter',
                ':auth: - AuthCenter', ':authright: - AuthRight',
                ':centrist: - Centrist'}
        # Drop na
        comments_df = comments_df[comments_df.author_flair_text.isin(labels)]

        comments_df.loc[comments_df['author_flair_text'].str.contains('Auth'), 'opinion'] = 1
        comments_df.loc[comments_df['author_flair_text'].str.contains('Lib'), 'opinion'] = -1
        comments_df.loc[comments_df['author_flair_text'].str.contains('Centrist'), 'opinion'] = 0
    else:
        raise Exception(axes + "Not implemented")



    # Create a lookup dictionary for scalar values
    opinion_dict = pd.Series(comments_df.opinion.values, index=comments_df.author).to_dict()

    # Step 3: Remove nodes not in comments_df or without "left" or "right" in author_flair_text
    nodes_to_remove = [node for node in G.nodes() if node not in opinion_dict]
    G.remove_nodes_from(nodes_to_remove)

    # Step 4: Build the array of scalar values for remaining nodes in the graph
    x = np.array([opinion_dict[node] for node in G.nodes()])

    return x, G

############################################################################
####################         PROPAGATION DATA           ####################
############################################################################

def compute_average_opinion(propagations, polarities, g):
    node2polarities = {}
    for prop_idx, active_nodes in enumerate(propagations):
        for node in active_nodes:
            if str(node) in node2polarities:
                node2polarities[str(node)].append(polarities[prop_idx])
            else:
                node2polarities[str(node)] = [polarities[prop_idx]]
    opinions = np.array([np.mean(node2polarities[username]) for username in g.nodes()])
    return opinions


def generate_time_ordered_pairs(prop):
    """Function to generate time-ordered pairs for one propagation list
       prop: Iterable : ordered list of active users (first to last in time)
       
       Returns pairs u->v (u activated before v, so u can influence v) for a given prop
    """
    pairs = []
    for i in range(len(prop)):
        for j in range(i+1, len(prop)):
            pairs.append((str(prop[i]), str(prop[j])))
    return pairs

def build_retweet_exposure_graph(propagations):
    """propagations: Iterable All propagations are list ordered by time (first active, to last active user)
       Returns influence graph given by exposure to previous content: 
       u --> v: u can influence v since it activated before v in a propagation.
    """
    prop_edge_list = [generate_time_ordered_pairs(prop) for prop in propagations]

    # Flatten the list of lists of pairs to get a single list of pairs
    all_pairs = [pair for sublist in prop_edge_list for pair in sublist]

    # Create an empty directed graph
    G = nx.DiGraph()

    # Add edges to the graph from the pairs
    for pair in all_pairs:
        G.add_edge(*pair) 
    return G
