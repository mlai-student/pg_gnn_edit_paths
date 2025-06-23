import copy
import os
from pathlib import Path
from typing import List, Any

import networkx as nx
import numpy as np
import torch


def get_k_lowest_nonzero_indices(tensor, k):
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Get the indices of non-zero elements
    non_zero_indices = torch.nonzero(flat_tensor, as_tuple=True)[0]

    # Select the non-zero elements
    non_zero_elements = torch.index_select(flat_tensor, 0, non_zero_indices)

    # Get the indices of the k lowest elements
    k_lowest_values, k_lowest_indices = torch.topk(non_zero_elements, k, largest=False)

    # Get the original indices
    k_lowest_original_indices = non_zero_indices[k_lowest_indices]

    return k_lowest_original_indices


def save_graphs(path: Path, db_name, graphs: List[nx.Graph], labels: List[int] = None, with_degree=False, graph_format=None):
    # save in two files DBName_Nodes.txt and DBName_Edges.txt
    # DBName_Nodes.txt has the following structure GraphId NodeId Feature1 Feature2 ...
    # DBName_Edges.txt has the following structure GraphId Node1 Node2 Feature1 Feature2 ...
    # DBName_Labels.txt has the following structure GraphId Label
    # if not folder db_name exists in path create it
    if not os.path.exists(path.joinpath(Path(db_name))):
        os.makedirs(path.joinpath(Path(db_name)))
    # create processed and raw folders in path+db_name
    if not os.path.exists(path.joinpath(Path(db_name + "/processed"))):
        os.makedirs(path.joinpath(Path(db_name + "/processed")))
    if not os.path.exists(path.joinpath(Path(db_name + "/raw"))):
        os.makedirs(path.joinpath(Path(db_name + "/raw")))
    # update path to write into raw folder
    path = path.joinpath(Path(db_name + "/raw/"))
    with open(path.joinpath(Path(db_name + "_Nodes.txt")), "w") as f:
        for i, graph in enumerate(graphs):
            for node in graph.nodes(data=True):
                # get list of all data entries of the node, first label then the rest
                if 'primary_label' not in node[1]:
                    data_list = [0]
                    if with_degree:
                        data_list.append(graph.degree(node[0]))
                elif type(node[1]['primary_label']) == np.ndarray or type(node[1]['primary_label']) == list:
                        data_list = [int(node[1]['primary_label'][0])]
                        for v in node[1]['primary_label'][1:]:
                            data_list.append(v)
                else:
                    data_list = [int(node[1]['primary_label'])]
                # append all the other features
                for key, value in node[1].items():
                    if key != 'primary_label':
                        if type(value) == int:
                            data_list.append(value)
                        elif type(value) == np.ndarray or type(value) == list:
                            for v in value:
                                data_list.append(v)
                f.write(str(i) + " " + str(node[0]) + " " + " ".join(map(str, data_list)) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()
    with open(path.joinpath(db_name + "_Edges.txt"), "w") as f:
        for i, graph in enumerate(graphs):
            for edge in graph.edges(data=True):
                # get list of all data entries of the node, first label then the rest
                if 'label' not in edge[2]:
                    data_list = [0]
                else:
                    if type(edge[2]['label']) == np.ndarray or type(edge[2]['label']) == list:
                        if len(edge[2]['label']) == 1:
                            data_list = [int(edge[2]['label'][0])]
                        else:
                            # raise an error as the label must be a single value
                            raise ValueError("Edge label must be a single value")
                    else:
                        data_list = [int(edge[2]['label'])]
                # append all the other features
                for key, value in edge[2].items():
                    if key != 'label':
                        if type(value) == int:
                            data_list.append(value)
                        elif type(value) == np.ndarray or type(value) == list:
                            for v in value:
                                data_list.append(v)
                f.write(str(i) + " " + str(edge[0]) + " " + str(edge[1]) + " " + " ".join(map(str, data_list)) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()
    if graph_format == 'NEL':
        with open(path.joinpath(db_name + "_Labels.txt"), "w") as f:
            if labels is not None:
                for i, label in enumerate(labels):
                    if type(label) == int:
                        f.write(db_name + " " + str(i) + " " + str(label) + "\n")
                    elif type(label) == np.ndarray or type(label) == list:
                        f.write(db_name + " " + str(i) + " " + " ".join(map(str, label)) + "\n")
                    else:
                        f.write(db_name + " " + str(i) + " " + str(label) + "\n")
            else:
                for i in range(len(graphs)):
                    f.write(db_name + " " + str(i) + " " + str(0) + "\n")
            # remove last empty line
            if f.tell() > 0:
                f.seek(f.tell() - 1, 0)
                f.truncate()
    else:
        with open(path.joinpath(db_name + "_Labels.txt"), "w") as f:
            if labels is not None:
                for i, label in enumerate(labels):
                    if type(label) == int:
                        f.write(str(i) + " " + str(label) + "\n")
                    elif type(label) == np.ndarray or type(label) == list:
                        f.write(str(i) + " " + " ".join(map(str, label)) + "\n")
                    else:
                        f.write(str(i) + " " + str(label) + "\n")
            else:
                for i in range(len(graphs)):
                    f.write(str(i) + " " + str(0) + "\n")
            # remove last empty line
            if f.tell() > 0:
                f.seek(f.tell() - 1, 0)
                f.truncate()


def load_graphs(path: Path, db_name: str, graph_format=None):
    graphs = []
    labels = []
    with open(path.joinpath(db_name + "_Nodes.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            node_id = int(data[1])
            feature = list(map(float, data[2:]))
            while len(graphs) <= graph_id:
                graphs.append(nx.Graph())
            graphs[graph_id].add_node(node_id, label=feature)
    with open(path.joinpath(db_name + "_Edges.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            node1 = int(data[1])
            node2 = int(data[2])
            feature = list(map(float, data[3:]))
            graphs[graph_id].add_edge(node1, node2, label=feature)
    if graph_format == 'NEL':
        with open(path.joinpath(db_name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = line.strip().split(" ")
                graph_name = data[0]
                graphs[i].name = graph_name
                graph_id = int(data[1])
                if len(data) == 3:
                    # first try to convert to int
                    try:
                        label = int(data[2])
                    except:
                        # if it fails convert to float
                        try:
                            label = float(data[2])
                        except:
                            # if it fails raise an error
                            raise ValueError("Label is not in the correct format")

                else:
                    label = list(map(float, data[2:]))

                while len(labels) <= graph_id:
                    labels.append(label)
                labels[graph_id] = label
    else:
        with open(path.joinpath(db_name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(" ")
                graph_id = int(data[0])
                if len(data) == 2:
                    # first try to convert to int
                    try:
                        label = int(data[1])
                    except:
                        # if it fails convert to float
                        try:
                            label = float(data[1])
                        except:
                            # if it fails raise an error
                            raise ValueError("Label is not in the correct format")

                else:
                    label = list(map(float, data[1:]))

                while len(labels) <= graph_id:
                    labels.append(label)
                labels[graph_id] = label
    return graphs, labels


def is_pruning(config:None) -> bool:
    if 'prune' in config and 'enabled' in config['prune'] and config['prune']['enabled']:
        return True
    return False




def reshape_indices(a, b):
    reshape_dict = {}
    ita = np.nditer(a, flags=['multi_index'])
    itb = np.nditer(b, flags=['multi_index'])
    while not ita.finished:
        reshape_dict[ita.multi_index] = itb.multi_index
        ita.iternext()
        itb.iternext()

    return reshape_dict


def convert_to_tuple(value: List):
    """
    Convert the value to a tuple and each element of the value to a tuple if it is a list
    """
    # create copy of value
    new_value = copy.deepcopy(value)
    for i, v in enumerate(new_value):
        if type(v) == list:
            new_value[i] = tuple(v)
    return tuple(new_value)


def convert_to_list(value: Any):
    if type(value) == int:
        return value
    elif type(value) == tuple:
        value = list(value)
        for i, v in enumerate(value):
            if type(v) == tuple:
                value[i] = list(v)
        return value



