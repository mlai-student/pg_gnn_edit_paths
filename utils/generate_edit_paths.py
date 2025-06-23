from math import inf
import concurrent.futures
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
    i, j, nx_graphs, db_name, loaded_edit_paths, optimization_iterations, timeout = params

    # check whether the loaded edit paths already contain the pair (i, j)
    if loaded_edit_paths is not None:
        if (i, j) in loaded_edit_paths:
            print(f"Edit path for graphs {i} and {j} already exist. Skipping generation.")
            return ((i, j), loaded_edit_paths[(i, j)])

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
                optimal_edit_paths = [EditPath(db_name, i, j, start_graph=nx_graphs[i], end_graph=nx_graphs[j], edit_path=x)]
                current_edit_distance = edit_distance_x
            print(f"Generated edit path {p+1} / {optimization_iterations} for graphs {i} and {j}")
            p += 1
        except StopIteration:
            print(f"Finished generating edit paths for graphs {i} and {j} because the timeout was reached or no more edit paths are available.")
            break

    return ((i, j), optimal_edit_paths)

def generate_pairwise_edit_paths(graph_dataset, db_name,
                                 output_dir:str = 'data/',
                                 optimization_iterations:int = 1,
                                 timeout:int = 1000000):
    """
    Generates pairwise optimal paths for the given database.

    Args:
        graph_dataset: GNNDataset object containing the dataset.
        db_name: Name of the database.
        output_dir: Directory where the edit paths will be saved.
        optimization_iterations: Number of optimization iterations to perform for each graph pair. TODO check if a value > 1 improves the length of the edit paths
    """

    nx_graphs = graph_dataset.nx_graphs
    if nx_graphs is None:
        raise ValueError("No NetworkX graphs found in the GraphDataset. Please create them first using graph_dataset.create_nx_graphs()")
    # plot the first graph
    plot_graph(nx_graphs[0], with_node_ids=True)
    # plot the second graph
    plot_graph(nx_graphs[1], with_node_ids=True)

    # Load existing edit paths
    loaded_edit_paths = load_edit_paths_from_file(db_name=db_name, file_path='data')

    # Prepare parameters for parallel processing
    graph_pairs = []
    for i in range(len(nx_graphs)):
        for j in range(i + 1, len(nx_graphs)):
            graph_pairs.append((i, j, nx_graphs, db_name, loaded_edit_paths, 
                               optimization_iterations, timeout))

    # Process graph pairs in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # split the graph pairs into chunks for parallel processing
        chunk_size = 100
        graph_pair_chunks = [graph_pairs[i:i + chunk_size] for i in range(0, len(graph_pairs), chunk_size)]
        for pairs in graph_pair_chunks:
            results = executor.map(process_graph_pair, pairs)
            global_edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=output_dir) if loaded_edit_paths is not None else dict()
            # Collect results
            for (pair_key, edit_paths) in results:
                global_edit_paths[pair_key] = edit_paths
            # Save after each pair is processed
            save_edit_path_to_file(db_name, global_edit_paths, file_path=output_dir)

    return
