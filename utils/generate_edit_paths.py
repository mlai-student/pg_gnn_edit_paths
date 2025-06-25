from math import inf
import concurrent.futures
from typing import List, Tuple

import networkx as nx
from utils.plotting import plot_graph
from utils.EditPath import EditPath
from utils.io import save_edit_path_to_file, load_edit_paths_from_file


def node_match_primary(n1, n2):
    """Match nodes based on their primary label."""
    return n1['primary_label'] == n2['primary_label']

def edge_match_primary(e1, e2):
    """Match edges based on their label."""
    return e1['label'] == e2['label']

def process_graph_pair(params):
    """
    Process a single pair of graphs to generate edit paths.

    Args:
        params: A tuple containing (i, j, nx_graphs, db_name, loaded_edit_paths, 
                optimization_iterations, timeout)

    Returns:
        A tuple of ((i, j), optimal_edit_paths) where optimal_edit_paths is a list of EditPath objects
    """
    i, j, nx_graphs, db_name, optimization_iterations, timeout = params

    # set current distance to infinity
    current_edit_distance = float('inf')
    print(f"Comparing graph {i} with graph {j}")
    # Use the global node_match_primary and edge_match_primary functions
    result_nx = nx.optimize_edit_paths(nx_graphs[i], nx_graphs[j], node_match=node_match_primary, edge_match=edge_match_primary, timeout=timeout)
    optimal_edit_paths = []
    p = 0
    while p < optimization_iterations:
        try:
            x = next(result_nx)
            edit_distance_x = x[2]
            if edit_distance_x < current_edit_distance:
                optimal_edit_paths = [EditPath(db_name, i, j, start_graph=nx_graphs[i], end_graph=nx_graphs[j], edit_path=x, iteration=p+1, max_iterations=optimization_iterations, timeout=timeout)]
                current_edit_distance = edit_distance_x
            print(f"Generated edit path {p+1} / {optimization_iterations} for graphs {i} and {j}")
            p += 1
        except StopIteration:
            print(f"Finished generating edit paths for graphs {i} and {j} because the timeout was reached or no more edit paths are available.")
            break

    return ((i, j), optimal_edit_paths)

def generate_pairwise_edit_paths(graph_dataset, db_name,
                                 output_dir:str = 'data/',
                                 missing_keys:List[Tuple[int, int]] = None,
                                 parameters:dict = None):
    """
    Generates pairwise optimal paths for the given database.

    Args:
        :param graph_dataset: GNNDataset object containing the dataset.
        :param db_name: Name of the database.
        :param output_dir: Directory where the edit paths will be saved.
        :param missing_keys: List of tuples containing the indices of the graph pairs for which edit paths should be generated.
        :param parameters: Dictionary containing the parameters for the edit path generation, i.e.,
            - optimization_iterations: Number of optimization iterations to perform for each graph pair. TODO check if a value > 1 improves the length of the edit paths
            - timeout: Maximum time in seconds for the optimization process.
            - max_workers: Maximum number of worker processes to use. If None, uses the number of processors on the machine.
    """

    nx_graphs = graph_dataset.nx_graphs
    if nx_graphs is None:
        raise ValueError("No NetworkX graphs found in the GraphDataset. Please create them first using graph_dataset.create_nx_graphs()")
    # plot the first graph
    plot_graph(nx_graphs[0], with_node_ids=True)
    # plot the second graph
    plot_graph(nx_graphs[1], with_node_ids=True)

    # Prepare parameters for parallel processing
    graph_pairs = []
    for i, j in missing_keys:
        graph_pairs.append((i, j, nx_graphs, db_name,
                               parameters["optimization_iterations"], parameters["timeout"]))

    # Process graph pairs in parallel
    with (concurrent.futures.ProcessPoolExecutor(max_workers=parameters["max_workers"]) as executor):
        # split the graph pairs into chunks for parallel processing
        chunk_size = 100
        graph_pair_chunks = [graph_pairs[i:i + chunk_size] for i in range(0, len(graph_pairs), chunk_size)]
        for pairs in graph_pair_chunks:
            # Load existing edit paths
            results = executor.map(process_graph_pair, pairs)
            global_edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=output_dir)
            if global_edit_paths is None:
                global_edit_paths = dict()
            # Collect results
            for (pair_key, edit_paths) in results:
                global_edit_paths[pair_key] = edit_paths
            # Save after each pair is processed
            save_edit_path_to_file(db_name, global_edit_paths, file_path=output_dir)
    return
