from utils.GraphLoader.GraphLoader import GraphDataset
from utils.generate_edit_paths import generate_pairwise_edit_paths
from utils.io import load_edit_paths_from_file
import os

if __name__ == '__main__':
    # set the parameters for the generation of the edit paths
    edit_path_generation_parameters = {'optimization_iterations': 100, 'timeout': 1, 'max_workers': None}


    # create data folder if it does not exist
    if not os.path.exists('data'):
        os.makedirs('data')
    db_name = 'MUTAG'
    # Load the Mutag dataset with the GNNDataset class
    graph_dataset = GraphDataset(
        root='data',
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
    print(f"Maximum runtime: {round(edit_path_generation_parameters['timeout'] * len(graph_dataset) * (len(graph_dataset) - 1) / 2 / (60 * 60), 2)} hours")
    print(f"Loaded MUTAG dataset with {len(graph_dataset)} graphs")
    # check wheter all edit paths already exist
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path='data')
    num_keys = 0
    # missing keys are all combinations
    missing_keys = [(i, j) for i in range(len(graph_dataset)) for j in range(i + 1, len(graph_dataset))]
    # get the number of keys in the edit paths dictionary
    if edit_paths is not None:
        num_keys = len(edit_paths.keys())
        missing_keys = [key for key in missing_keys if key not in edit_paths]
    if num_keys == len(graph_dataset) * (len(graph_dataset) - 1) // 2:
        print(f"All edit paths already exist in the file data/{db_name}_ged_paths.paths. Skipping generation.")
    else:
        # generate the pairwise edit paths (not optimal) and save them to a file
        generate_pairwise_edit_paths(graph_dataset,
                                     db_name=db_name,
                                     missing_keys=missing_keys,
                                     output_dir='data/',
                                     optimization_iterations=edit_path_generation_parameters['optimization_iterations'],
                                     timeout=edit_path_generation_parameters['timeout'],
                                     max_workers=edit_path_generation_parameters['max_workers'])
    # get the graphs of the dataset
    nx_graphs = graph_dataset.nx_graphs
    # load the edit paths from the file
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path='data')
    # example of creating edit path graphs using some valid random permutation of the operations
    # take the edit path with index 0 between the first two graphs (0, 1)
    edit_path_graphs = edit_paths[(0, 1)][0].create_edit_path_graphs(nx_graphs[0], nx_graphs[1], seed=42)
