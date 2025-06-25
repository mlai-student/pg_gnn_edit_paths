from utils.GraphLoader.GraphLoader import GraphDataset
from utils.generate_edit_paths import generate_pairwise_edit_paths
from utils.io import load_edit_paths_from_file
import os
import argparse
import multiprocessing

def main(db_name='MUTAG', optimization_iterations=100, timeout=60, max_workers=None, output_dir='data'):
    """
    Main function to generate and process edit paths for graph datasets.

    Args:
        db_name (str): Name of the database/dataset to use
        optimization_iterations (int): Number of optimization iterations
        timeout (int): Timeout in seconds for each graph pair processing
        max_workers (int): Maximum number of worker processes, None for auto-detection
        output_dir (str): Directory to store output files
    """
    # set the parameters for the generation of the edit paths
    edit_path_generation_parameters = {
        'optimization_iterations': optimization_iterations, 
        'timeout': timeout, 
        'max_workers': max_workers
    }

    # get max number of workers if max_workers is set to None
    if edit_path_generation_parameters['max_workers'] is None:
        edit_path_generation_parameters['max_workers'] = multiprocessing.cpu_count()

    # create data folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load the dataset with the GNNDataset class
    graph_dataset = GraphDataset(
        root=output_dir,
        name=db_name,
        from_existing_data='TUDataset',
        task='graph_classification'
    )
    # create the nx_graphs from the pytorch geometric dataset
    graph_dataset.create_nx_graphs()
    # print the maximum runtime for the generation of the edit paths
    print(f"Generating pairwise edit paths for the {db_name} dataset with the following parameters: {edit_path_generation_parameters}")
    # print the number of graphs in the dataset
    print(f"Number of graphs in the {db_name} dataset: {len(graph_dataset)}")
    # print the maximum time
    print(f"Maximum runtime: {round(edit_path_generation_parameters['timeout'] * len(graph_dataset) * (len(graph_dataset) - 1) / 2 / (60 * 60), 2) / edit_path_generation_parameters['max_workers']} hours")
    print(f"Loaded {db_name} dataset with {len(graph_dataset)} graphs")
    # check whether all edit paths already exist
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=output_dir)
    num_keys = 0
    # missing keys are all combinations
    missing_keys = [(i, j) for i in range(len(graph_dataset)) for j in range(i + 1, len(graph_dataset))]
    # get the number of keys in the edit paths dictionary
    if edit_paths is not None:
        num_keys = len(edit_paths.keys())
        missing_keys = [key for key in missing_keys if key not in edit_paths]
    if num_keys == len(graph_dataset) * (len(graph_dataset) - 1) // 2:
        print(f"All edit paths already exist in the file {output_dir}/{db_name}_ged_paths.paths. Skipping generation.")
    else:
        # generate the pairwise edit paths (not optimal) and save them to a file
        generate_pairwise_edit_paths(graph_dataset,
                                     db_name=db_name,
                                     output_dir=f"{output_dir}/",
                                     missing_keys=missing_keys,
                                     parameters=edit_path_generation_parameters)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate and process edit paths for graph datasets.')
    parser.add_argument('--db_name', type=str, default='MUTAG',
                        help='Name of the database/dataset to use (default: MUTAG)')
    parser.add_argument('--optimization_iterations', type=int, default=100,
                        help='Number of optimization iterations (default: 100)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds for each graph pair processing (default: 60)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker processes (default: None for auto-detection)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to store output files (default: data)')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(
        db_name=args.db_name,
        optimization_iterations=args.optimization_iterations,
        timeout=args.timeout,
        max_workers=args.max_workers,
        output_dir=args.output_dir
    )
