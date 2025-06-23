from utils.GraphLoader.GraphLoader import GraphDataset
from utils.generate_edit_paths import generate_pairwise_edit_paths
from utils.io import load_edit_paths_from_file
import os

if __name__ == '__main__':
    # set the parameters for the generation of the edit paths
    edit_path_generation_parameters = {'optimization_iterations': 100,
                                       'max_num_edit_paths': 5,
                                       'timeout': 10}


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
    print(f"Loaded MUTAG dataset with {len(graph_dataset)} graphs")
    # generate the pairwise edit paths (not optimal) and save them to a file
    generate_pairwise_edit_paths(graph_dataset,
                                 db_name=db_name,
                                 output_dir='data/',
                                 optimization_iterations=edit_path_generation_parameters['optimization_iterations'],
                                 max_num_edit_paths=edit_path_generation_parameters['max_num_edit_paths'],
                                    timeout=edit_path_generation_parameters['timeout'])
    # get the graphs of the dataset
    nx_graphs = graph_dataset.nx_graphs
    # load the edit paths from the file
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path='data')
    # example of creating edit path graphs using some valid random permutation of the operations
    # take the edit path with index 0 between the first two graphs (0, 1)
    edit_path_graphs = edit_paths[(0, 1)][0].create_edit_path_graphs(nx_graphs[0], nx_graphs[1], seed=42)
