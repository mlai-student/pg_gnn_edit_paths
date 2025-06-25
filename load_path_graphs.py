from utils.GraphLoader.GraphLoader import GraphDataset
from utils.io import load_edit_paths_from_file
import argparse


def load_edit_path_between_graphs(graph_dataset: GraphDataset, db_name: str, data_dir: str, start_graph_id: int, end_graph_id: int):
    """
    Load edit path between two graphs and create edit path graphs.

    Args:
        graph_dataset (GraphDataset): The graph dataset containing the graphs
        db_name (str): Name of the database/dataset
        data_dir (str): Directory where the edit paths are stored
        start_graph_id (int): ID of the start graph
        end_graph_id (int): ID of the end graph

    Returns:
        list: List of edit path graphs
    """
    # get the graphs of the dataset
    nx_graphs = graph_dataset.nx_graphs
    if nx_graphs is None or len(nx_graphs) == 0:
        raise ValueError("No graphs found in the dataset. Please create the graphs first.")
    # load the edit paths from the file
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=data_dir)
    # example of creating edit path graphs using some valid random permutation of the operations
    # take the edit path with index 0 between the graphs with the given ids
    edit_path_graphs = edit_paths[(start_graph_id, end_graph_id)][0].create_edit_path_graphs(nx_graphs[start_graph_id], nx_graphs[end_graph_id], seed=42)

    return edit_path_graphs


def main(db_name='MUTAG', data_dir='data', start_graph_id=0, end_graph_id=1):
    """
    Main function to load edit paths between graphs and create edit path graphs.

    Args:
        db_name (str): Name of the database/dataset to use
        data_dir (str): Directory where the edit paths are stored
        start_graph_id (int): ID of the start graph
        end_graph_id (int): ID of the end graph

    Returns:
        list: List of edit path graphs
    """
    # Load the dataset with the GNNDataset class
    graph_dataset = GraphDataset(
        root=data_dir,
        name=db_name,
        from_existing_data='TUDataset',
        task='graph_classification'
    )
    # create the nx graphs
    graph_dataset.create_nx_graphs()

    # Load edit path between graphs
    edit_path_graphs = load_edit_path_between_graphs(
        graph_dataset, 
        db_name, 
        data_dir, 
        start_graph_id=start_graph_id, 
        end_graph_id=end_graph_id
    )

    return edit_path_graphs


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Load edit paths between graphs and create edit path graphs.')
    parser.add_argument('--db_name', type=str, default='MUTAG',
                        help='Name of the database/dataset to use (default: MUTAG)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory where the edit paths are stored (default: data)')
    parser.add_argument('--start_graph_id', type=int, default=0,
                        help='ID of the start graph (default: 0)')
    parser.add_argument('--end_graph_id', type=int, default=1,
                        help='ID of the end graph (default: 1)')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    edit_path_graphs = main(
        db_name=args.db_name,
        data_dir=args.data_dir,
        start_graph_id=args.start_graph_id,
        end_graph_id=args.end_graph_id
    )

    print(f"Successfully loaded edit path graphs between graph {args.start_graph_id} and {args.end_graph_id} from {args.db_name} dataset.")
