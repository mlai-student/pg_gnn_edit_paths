import networkx as nx
from utils.plotting import plot_graph
from utils.EditPath import EditPath
from utils.io import save_edit_path_to_file


def generate_pairwise_optimal_paths(graph_dataset, db_name, output_dir:str = 'data/'):
    """
    Generates pairwise optimal paths for the given database.

    Args:
        graph_dataset: GNNDataset object containing the dataset.
        db_name: Name of the database.
        output_dir: Directory where the edit paths will be saved.
    """

    # TODO check whether node match and edge match are necessary
    def node_match_primary(n1, n2):
        return n1['primary_label'] == n2['primary_label']

    def edge_match_primary(e1, e2):
        return e1['label'] == e2['label']

    # TODO check if a value > 1 improves the length of the edit paths
    num_optimization_iterations = 1
    nx_graphs = graph_dataset.nx_graphs
    if nx_graphs is None:
        raise ValueError("No NetworkX graphs found in the GraphDataset. Please create them first using graph_dataset.create_nx_graphs()")
    # plot the first graph
    plot_graph(nx_graphs[0], with_node_ids=True)
    # plot the second graph
    plot_graph(nx_graphs[1], with_node_ids=True)
    # iterate over all the graph pairs
    global_edit_paths = dict()
    for i in range(len(nx_graphs)):
        for j in range(i + 1, len(nx_graphs)):
            print(f"Comparing graph {i} with graph {j}")
            result_nx = nx.optimize_edit_paths(nx_graphs[i], nx_graphs[j], node_match=node_match_primary, edge_match=edge_match_primary)
            optimal_edit_paths = []
            p = 0
            while p < num_optimization_iterations:
                x = next(result_nx)
                optimal_edit_paths.append(EditPath(db_name, i, j, start_graph=nx_graphs[i], end_graph=nx_graphs[j], edit_path=x))
                print(f"Generated edit path {p+1} / {num_optimization_iterations} for graphs {i} and {j}")
                p += 1
            global_edit_paths[(i, j)] = optimal_edit_paths
    save_edit_path_to_file(db_name, global_edit_paths, file_path=output_dir)
    return
