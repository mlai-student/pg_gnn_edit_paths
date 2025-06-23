import os
from pathlib import Path
from typing import Dict, Optional, Callable, List, Union

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import ZINC, TUDataset, GNNBenchmarkDataset, LRGBDataset

from utils.GraphLoader.GraphLabels import NodeLabels, EdgeLabels

from torch_geometric.io import fs
from torch_geometric.utils.convert import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset


class GraphDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            from_existing_data: Union[str, list, None] = None,
            force_reload: bool = False,
            use_node_attr: bool = True,
            use_edge_attr: bool = True,
            delete_zero_columns: bool = True,
            precision: str = 'float',
            input_features = None,
            output_features = None,
            task = None,
            merge_graphs = None,
            testing: Optional[int] = None,
    ) -> None:
        self.name = name # name of the dataset
        self.from_existing_data = from_existing_data # create the dataset from existing data
        self.nx_graphs = None # networkx graphs
        self.unique_node_labels = 0 # number of unique node labels
        self.node_labels = {} # different node labels for the graph data
        self.edge_labels = {} # different edge labels for the graph data
        self.properties = {} # different pairwise properties for the graph data
        self.precision = torch.float
        self.task = task
        self.testing = testing # take the first n graphs for testing (only for debugging)
        if precision == 'double':
            self.precision = torch.double
        super(GraphDataset, self).__init__(root, transform, pre_transform, force_reload=force_reload)
        out = fs.torch_load(self.processed_paths[0])
        if testing is not None:
            if isinstance(testing, int):
                # reduce out to the first testing graphs
                # slices
                testing_slices = {'edge_index': out[1]['edge_index'][:testing],
                'edge_attr': out[1]['edge_attr'][:testing],
                'x': out[1]['x'][:testing],
                'y': out[1]['y'][:testing]}
                testing_data = {
                    'num_nodes': testing_slices['y'][-1].item(),
                    'edge_index': out[0]['edge_index'][:, :testing_slices['edge_index'][-1]],
                    'edge_attr': out[0]['edge_attr'][:testing_slices['edge_attr'][-1], :],
                    'x': out[0]['x'][:testing_slices['x'][-1]],
                    'y': out[0]['y'][:testing_slices['y'][-1]],
                }
                testing_sizes = {
                    'num_node_labels': out[2]['num_node_labels'],
                    'num_node_attributes': out[2]['num_node_attributes'],
                    'num_edge_labels': out[2]['num_edge_labels'],
                    'num_edge_attributes': out[2]['num_edge_attributes'],
                }

            else:
                raise ValueError("testing must be an integer or None")

        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if testing is not None:
            data, self.slices, self.sizes, data_cls = testing_data, testing_slices, testing_sizes, out[3]
        else:
            if len(out) == 3:  # Backward compatibility.
                data, self.slices, self.sizes = out
                data_cls = Data
            else:
                data, self.slices, self.sizes, data_cls = out

        self._num_graph_nodes = torch.zeros(len(self), dtype=torch.long)
        num_node_attributes = self.num_node_attributes
        num_edge_attributes = self.num_edge_attributes

        # split node labels and attributes as well as edge labels and attributes
        num_zero_columns = 0
        if data.get('x', None) is not None:
            if delete_zero_columns and data['x'].layout != torch.sparse_csr:
                # if x is one dimensional, add a dimension
                if data['x'].dim() == 1:
                    data['x'] = data['x'].unsqueeze(1)
                columns = data['x'].shape[1]
                # remove columns with only zeros
                if self.precision == torch.float:
                    data['x'] = data['x'][:, data['x'].sum(dim=0) != 0].float()
                else:
                    data['x'] = data['x'][:, data['x'].sum(dim=0) != 0].double()
                num_zero_columns = columns - data['x'].shape[1]
                if self.task == 'graph_classification' or self.task == 'graph_regression':
                    self.sizes['num_node_labels'] = data['x'].shape[1]
        else:
            if data.get('num_nodes', None) is None:
                data['num_nodes'] = torch.zeros(len(self), dtype=torch.long)
            # create data['x'] using vectors of ones
            data['x'] = torch.ones(data['num_nodes'], 1, dtype=self.precision)
            self.sizes['num_node_labels'] = 1
            self.slices['x'] = [0]
            self.slices['x'] += data['_num_nodes']
            self.slices['x'] = torch.tensor(self.slices['x'], dtype=torch.long).cumsum(dim=0)

        if len(data['x'].shape) == 1:
            data['x'] = data['x'].unsqueeze(1)
        if data['x'].shape[1] == 1:
            self.node_labels['primary'] = data['x'].clone().detach().long()
        else:
            if self.task == 'graph_classification' or self.task == 'graph_regression':
                if data['x'].shape[1] - (num_node_attributes - num_zero_columns) == 1:
                    self.node_labels['primary'] = data['x'][:, -1].clone().detach().long()
                else:
                    self.node_labels['primary'] = torch.argmax(data['x'][:, num_node_attributes:], dim=1)
            elif self.task == 'node_classification':
                self.node_labels['primary'] = data['y']
            else:
                if data['x'].shape[1] - (num_node_attributes - num_zero_columns) == 1:
                    self.node_labels['primary'] = data['x'][:, -1].clone().detach().long()
                else:
                    self.node_labels['primary'] = torch.argmax(data['x'][:, num_node_attributes:], dim=1)
        self.unique_node_labels = torch.unique(self.node_labels['primary']).shape[0]
        if not use_node_attr:
            data['x'] = data['x'][:, self.num_node_attributes-num_zero_columns:]
            self.sizes['num_node_attributes'] = 0



        if data.get('edge_attr', None) is not None:
            if len(data['edge_attr'].shape) == 1:
                # unsqueeze the edge_attr tensor
                data['edge_attr'] = data['edge_attr'].unsqueeze(1)
            if data['edge_attr'].shape[1] == 1:
                self.edge_labels['primary'] = data['edge_attr'].clone().detach().long()
            else:
                if data['edge_attr'].shape[1] - num_edge_attributes == 1:
                    self.edge_labels['primary'] = data['edge_attr'][:, -1].clone().detach().long()
                else:
                    self.edge_labels['primary'] = torch.argmax(data['edge_attr'][:, num_edge_attributes:], dim=1)
            if not use_edge_attr:
                data['edge_attr'] = data['edge_attr'][:, self.num_edge_attributes:]
                self.sizes['num_edge_attributes'] = 0

        if data.get('y', None) is not None:
            if self.task == 'graph_classification':
                # convert y to long
                data['y'] = data['y'].long()
                # flatten y
                data['y'] = data['y'].view(-1)


        if len(self) == 1:
            data['num_nodes'] = torch.tensor([data['x'].shape[0]], dtype=torch.long)
        else:
            data['num_nodes'] = torch.zeros(len(self), dtype=torch.long)
            for i in range(len(self)):
                data['num_nodes'][i] = data['x'][self.slices['x'][i]:self.slices['x'][i+1]].shape[0]


        self.preprocess_share_gnn_data(data, input_features, output_features, task=task)

        self.number_of_output_classes = 0
        # use the task to determine the number of classes
        if self.task == 'graph_classification':
            self.number_of_output_classes = torch.unique(data['y']).shape[0]
        elif self.task == 'graph_regression':
            # if y is 2D-tensor, take the dimension of the second axis as number of output classes
            if data['y'].dim() == 2:
                self.number_of_output_classes = data['y'].shape[1]
            else:
                self.number_of_output_classes = 1
        elif self.task == 'node_classification':
            self.number_of_output_classes = self.num_node_labels
        elif self.task == 'edge_classification':
            self.number_of_output_classes = self.num_edge_labels
        elif self.task == 'link_prediction':
            self.number_of_output_classes = 2
        else:
            raise ValueError('Task not supported')


        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            # split node labels and attributes as well as edge labels and attributes
            self.data = data_cls.from_dict(data)


        assert isinstance(self._data, Data)

    @property
    def raw_dir(self) -> str:
        name = f'raw'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_classes(self) -> int:
        return self.number_of_output_classes


    def process(self):
        sizes = None
        if self.from_existing_data is not None:
            # merge the graphs
            if isinstance(self.from_existing_data, list):
                all_x = None
                all_edge_indices = None
                all_edge_atr = None
                all_y = None
                all_num_nodes = None
                self.slices = {
                    'x': [0],
                    'edge_index': [0],
                    'edge_attr': [0],
                    'y': [0]
                }
                sizes = {
                    'num_node_labels': 0,
                    'num_node_attributes': 0,
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }
                for i, dataset in enumerate(self.from_existing_data):
                    current_x = dataset.data.x
                    current_edge_indices = dataset.data.edge_index
                    current_edge_atr = dataset.data.edge_attr
                    current_y = dataset.data.y
                    current_num_nodes = dataset.data.num_nodes
                    if i == 0:
                        all_x = current_x
                        all_edge_indices = current_edge_indices
                        all_edge_atr = current_edge_atr
                        all_y = current_y
                        all_num_nodes = current_num_nodes
                        self.slices = {
                            'x': dataset.slices['x'],
                            'edge_index': dataset.slices['edge_index'],
                            'y': dataset.slices['y'],
                            'names': [dataset.name] * len(dataset)
                        }
                        if 'edge_attr' in dataset.slices:
                            self.slices['edge_attr'] = dataset.slices['edge_attr']
                        sizes = {
                            'num_node_labels': dataset.num_node_labels,
                            'num_node_attributes': dataset.num_node_attributes,
                            'num_edge_labels': dataset.num_edge_labels,
                            'num_edge_attributes': dataset.num_edge_attributes,
                        }
                    else:
                        max_node_labels = max(sizes['num_node_labels'], dataset.num_node_labels)
                        max_node_attrs = max(sizes['num_node_attributes'], dataset.num_node_attributes)
                        max_edge_labels = max(sizes['num_edge_labels'], dataset.num_edge_labels)
                        max_edge_attrs = max(sizes['num_edge_attributes'], dataset.num_edge_attributes)

                        self.slices['x'] =  torch.cat((self.slices['x'], dataset.slices['x'][1:] + all_x.shape[0]), dim=0)
                        self.slices['edge_index'] = torch.cat((self.slices['edge_index'], dataset.slices['edge_index'][1:] + all_edge_indices.shape[1]), dim=0)
                        if 'edge_attr' in self.slices:
                            self.slices['edge_attr'] = torch.cat((self.slices['edge_attr'], dataset.slices['edge_attr'][1:] + all_edge_atr.shape[0]), dim=0)
                        self.slices['y'] = torch.cat((self.slices['y'], dataset.slices['y'][1:] + all_y.shape[0]), dim=0)

                        # bring all tensors to the same size
                        if sizes['num_node_labels'] < max_node_labels:
                            # add zeros from sizes['num_node_labels'] to max_node_labels
                            all_x = torch.cat((all_x[:, :sizes['num_node_labels']], torch.zeros(all_x.shape[0], max_node_labels - sizes['num_node_labels']), all_x[:, sizes['num_node_labels']:]), dim=1)
                        if dataset.num_node_labels < max_node_labels:
                            current_x = torch.cat((current_x[:, :dataset.num_node_labels], torch.zeros(current_x.shape[0], max_node_labels - dataset.num_node_labels), current_x[:, dataset.num_node_labels:]), dim=1)
                        if sizes['num_node_attributes'] < max_node_attrs:
                            all_x = torch.cat((all_x, torch.zeros(all_x.shape[0], max_node_attrs - sizes['num_node_attributes'])), dim=1)
                        if dataset.num_node_features < max_node_attrs:
                            current_x = torch.cat((current_x, torch.zeros(current_x.shape[0], max_node_attrs - dataset.num_node_features)), dim=1)
                        all_x = torch.cat((all_x, current_x), dim=0)
                        if sizes['num_edge_labels'] < max_edge_labels and all_edge_atr is not None:
                            all_edge_atr = torch.cat((all_edge_atr[:, :sizes['num_edge_labels']], torch.zeros(all_edge_atr.shape[0], max_edge_labels - sizes['num_edge_labels']), all_edge_atr[:, sizes['num_edge_labels']:]), dim=1)
                        if dataset.num_edge_labels < max_edge_labels:
                            current_edge_atr = torch.cat((current_edge_atr[:, :dataset.num_edge_labels], torch.zeros(current_edge_atr.shape[0], max_edge_labels - dataset.num_edge_labels), current_edge_atr[:, dataset.num_edge_labels:]), dim=1)
                        if sizes['num_edge_attributes'] < max_edge_attrs:
                            all_edge_atr = torch.cat((all_edge_atr, torch.zeros(all_edge_atr.shape[0], max_edge_attrs - sizes['num_edge_attributes'])), dim=1)
                        if dataset.num_edge_attributes < max_edge_attrs:
                            current_edge_atr = torch.cat((current_edge_atr, torch.zeros(current_edge_atr.shape[0], max_edge_attrs - dataset.num_edge_attributes)), dim=1)
                        if 'edge_attr' in self.slices:
                            all_edge_atr = torch.cat((all_edge_atr, current_edge_atr), dim=0)
                            all_edge_atr = torch.cat((all_edge_atr, current_edge_atr), dim=0)
                        all_edge_indices = torch.cat((all_edge_indices, current_edge_indices), dim=1)
                        all_y = torch.cat((all_y, current_y), dim=0)
                        all_num_nodes = torch.cat((all_num_nodes, current_num_nodes), dim=0)

                        sizes['num_node_labels'] = max_node_labels
                        sizes['num_node_attributes'] = max_node_attrs
                        sizes['num_edge_labels'] = max_edge_labels
                        sizes['num_edge_attributes'] = max_edge_attrs
                        # make self data from all_x, all_edge_indices, all_edge_atr, all_y
                        self.data = Data(x=all_x, edge_index=all_edge_indices, edge_attr=all_edge_atr, y=all_y, num_nodes=all_num_nodes)




            elif self.from_existing_data in ['ZINC', 'ZINC-full', 'ZINC-Full', 'ZINCFull', 'ZINC-12k', 'ZINC-25k']:
                subset = True
                if self.name in ['ZINC-full', 'ZINC-Full', 'ZINCFull', 'ZINC-250k']:
                    subset = False
                train_data = ZINC(root='tmp/', subset=subset, split='train')
                validation_data = ZINC(root='tmp/', subset=subset, split='val')
                test_data = ZINC(root='tmp/', subset=subset, split='test')
                # merge train_data._data, validation_data._data and test_data._data
                all_data = torch_geometric.data.InMemoryDataset.collate([train_data._data, validation_data._data, test_data._data])

                self.data = all_data[0]
                # merge the slices
                self.slices = dict()
                for key in train_data.slices.keys():
                    validation_data.slices[key] += train_data.slices[key][-1]
                    test_data.slices[key] += validation_data.slices[key][-1]
                    self.slices[key] = torch.cat((train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

                sizes = {'num_edge_attributes': 0,
                         'num_edge_labels': len(torch.unique(self.data.edge_attr)),
                         'num_node_attributes': 0,
                         'num_node_labels': 0
                }
            elif self.from_existing_data == 'OGB_GraphProp':
                dataset_ogb = PygGraphPropPredDataset(name=self.name, root='tmp/')
                split_idx = dataset_ogb.get_idx_split()
                train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
                self.data = dataset_ogb.data
                # put column 0 of x and edge_attr to the end
                self.data.x = torch.cat((self.data.x[:, 1:], self.data.x[:, 0].unsqueeze(1)), dim=1)
                unique_node_labels = torch.unique(self.data.x[:, -1])
                # map unique node labels to integers 0, 1, 2, ...
                self.data.x[:, -1] = torch.tensor([torch.where(unique_node_labels == x)[0] for x in self.data.x[:, -1]], dtype=torch.long)
                self.data.edge_attr = torch.cat((self.data.edge_attr[:, 1:], self.data.edge_attr[:, 0].unsqueeze(1)), dim=1)
                unique_edge_labels = torch.unique(self.data.edge_attr[:, -1])
                # map unique edge labels to integers 0, 1, 2, ...
                self.data.edge_attr[:, -1] = torch.tensor([torch.where(unique_edge_labels == x)[0] for x in self.data.edge_attr[:, -1]], dtype=torch.long)
                self.slices = dataset_ogb.slices
                sizes = {
                    'num_node_labels': len(unique_node_labels),
                    'num_node_attributes': 8,
                    'num_edge_labels': len(unique_edge_labels),
                    'num_edge_attributes': 2,
                }
                pass
            elif self.from_existing_data == 'MoleculeNet':
                dataset = torch_geometric.datasets.MoleculeNet(root='tmp/', name=self.name)
                self.data = dataset.data
                # put column 0 of x and edge_attr to the end
                self.data.x = torch.cat((self.data.x[:, 1:], self.data.x[:, 0].unsqueeze(1)), dim=1)
                unique_node_labels = torch.unique(self.data.x[:, -1])
                # map unique node labels to integers 0, 1, 2, ...
                self.data.x[:, -1] = torch.tensor([torch.where(unique_node_labels == x)[0] for x in self.data.x[:, -1]], dtype=torch.long)
                self.data.edge_attr = torch.cat((self.data.edge_attr[:, 1:], self.data.edge_attr[:, 0].unsqueeze(1)), dim=1)
                unique_edge_labels = torch.unique(self.data.edge_attr[:, -1])
                # map unique edge labels to integers 0, 1, 2, ...
                self.data.edge_attr[:, -1] = torch.tensor([torch.where(unique_edge_labels == x)[0] for x in self.data.edge_attr[:, -1]], dtype=torch.long)
                self.slices = dataset.slices
                sizes = {
                    'num_node_labels': torch.unique(dataset.data.x[:,0]),
                    'num_node_attributes': 8,
                    'num_edge_labels': len(torch.unique(dataset.data.edge_attr[:, 0])),
                    'num_edge_attributes': 2,
                }
                pass
            elif self.from_existing_data == 'SubstructureBenchmark':
                # relative path to project root
                root = Path(__file__).resolve().parent.parent.parent
                train_data = GraphCount(root=str(root.joinpath('tmp')) + '/', split="train", task=self.name)
                validation_data = GraphCount(root=str(root.joinpath('tmp')) + '/', split="val", task=self.name)
                test_data = GraphCount(root=str(root.joinpath('tmp')) + '/', split="test", task=self.name)
                all_data = torch_geometric.data.InMemoryDataset.collate([train_data._data, validation_data._data, test_data._data])
                self.data = all_data[0]
                # flatten y if self.name is not 'substructure_counting'
                if self.name != 'substructure_counting':
                    self.data.y = self.data.y.view(-1)
                # merge the slices
                self.slices = dict()
                for key in train_data.slices.keys():
                    validation_data.slices[key] += train_data.slices[key][-1]
                    test_data.slices[key] += validation_data.slices[key][-1]
                    self.slices[key] = torch.cat((train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

                sizes = {'num_edge_attributes': 0,
                         'num_edge_labels': 0,
                         'num_node_attributes': 0,
                         'num_node_labels': 0
                }


                pass
            elif self.from_existing_data in ['planetoid', 'cora', 'citeseer', 'pubmed', 'Planetoid']:
                dataset = torch_geometric.datasets.Planetoid(root='tmp/', name=self.name)
                self.data = dataset[0]
                self.slices = dict()
                for key, value in dataset.data:
                    self.slices[key] = torch.tensor([0, value.shape[0]], dtype=torch.long)
                sizes = {
                    'num_node_labels': len(torch.unique(self.data.y)),
                    'num_node_attributes': self.data.x.shape[1],
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }
            elif self.from_existing_data in ['Nell', 'nell', 'NELL']:
                dataset = torch_geometric.datasets.NELL(root='tmp/')
                self.data = dataset[0]
                self.slices = dict()
                for key, value in dataset.data:
                    self.slices[key] = torch.tensor([0, value.shape[0]], dtype=torch.long)
                sizes = {
                    'num_node_labels': len(torch.unique(self.data.y)),
                    'num_node_attributes': self.data.x.shape[1],
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }
            elif self.from_existing_data in ['ogbn', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']:
                dataset = PygNodePropPredDataset(name=self.name, root='tmp/')
                split_idx = dataset.get_idx_split()
                train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
                self.data = dataset[0]  # pyg graph object
                self.data.train_mask = self.data.val_mask = self.data.test_mask = None
                self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                self.data.train_mask[train_idx] = 1
                self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                self.data.val_mask[valid_idx] = 1
                self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                self.data.test_mask[test_idx] = 1
                self.slices = dict()
                for key, value in self.data:
                    if isinstance(value, torch.Tensor):
                        self.slices[key] = torch.tensor([0, value.shape[0]], dtype=torch.long)
                sizes = {
                    'num_node_labels': len(torch.unique(self.data.y)),
                    'num_node_attributes': self.data.x.shape[1],
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }

            elif self.from_existing_data == 'TUDataset':
                tu_dataset = TUDataset(root='tmp/', name=self.name, use_node_attr=True, use_edge_attr=True)
                self.data, self.slices, sizes = tu_dataset._data, tu_dataset.slices, tu_dataset.sizes
            elif self.from_existing_data == 'NEL':
                self.data, self.slices, sizes = self.read_nel_data_v2()
            elif self.from_existing_data == 'gnn_benchmark':
                dataset = GNNBenchmarkDataset("tmp/", self.name)
                sizes = {
                    'num_node_labels': dataset.num_features,
                    'num_node_attributes': dataset.num_node_features,
                    'num_edge_labels': dataset.num_edge_features,
                    'num_edge_attributes': 0
                }
                # add x to data uing num_nodes times 0
                num_graphs = len(dataset.data.y)
                dataset.data.x = torch.ones(dataset.data.num_nodes, 1)
                nodes_per_graph = dataset.data.num_nodes // num_graphs
                # remove num_nodes from x
                dataset.slices['x'] = torch.linspace(0, dataset.data.num_nodes, num_graphs + 1, dtype=torch.long)
                self.slices = dataset.slices
                self.data = dataset.data
                pass
            elif self.from_existing_data == 'Peptides':
                dataset = torch_geometric.datasets.LRGBDataset(root='tmp/', name=self.name)
                self.data = dataset.data
                self.slices = dataset.slices
                sizes = {
                    'num_node_labels': dataset.num_node_features,
                    'num_node_attributes': dataset.num_node_features,
                    'num_edge_labels': dataset.num_edge_features,
                    'num_edge_attributes': dataset.num_edge_features
                }
        else:
            print('Cannot process the data')

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

    def read_nel_data(self):
        graphs, labels = load_graphs(Path(self.raw_dir), self.name, graph_format='NEL')
        node_labels = []
        node_attributes = []
        node_slices = [0]
        with_node_attributes = False
        for graph_id, graph in enumerate(graphs):
            print(f'Processing graph {graph_id+1}/{len(graphs)}')
            node_labels += [0] * graph.number_of_nodes()
            node_attributes += [0] * graph.number_of_nodes()
            for node in graph.nodes(data=True):
                if 'label' in node[1]:
                    index_start = np.sum(node_slices[0:graph_id+1])
                    node_labels[index_start+node[0]] = int(node[1]['label'][0])
                    if len(node[1]['label']) > 1:
                        with_node_attributes = True
                        node_attributes[index_start+node[0]] = node[1]['label'][1:]
            node_slices.append(graph.number_of_nodes())
        # convert the node labels to a tensor
        node_labels = torch.tensor(node_labels, dtype=torch.long)
        # apply row-wise one-hot encoding
        node_labels = torch.nn.functional.one_hot(node_labels).float()
        # convert the node attributes to a tensor
        node_attributes = torch.tensor(node_attributes, dtype=torch.float)
        if len(node_attributes) == 0 or not with_node_attributes:
            node_attributes = None
        if node_attributes is not None:
            # stack node attributes and node labels together to form the node feature matrix
            x = torch.cat((node_attributes, node_labels), dim=1)
        else:
            x = node_labels
        node_slices = torch.tensor(node_slices, dtype=torch.long).cumsum(dim=0)
        # create edge_index tensor
        edge_indices = []
        edge_slices = [0]
        edge_labels = []
        edge_attributes = []
        for i, graph in enumerate(graphs):
            for edge in graph.edges(data=True):
                edge_indices.append([edge[0], edge[1]])
                if 'label' in edge[2]:
                    edge_labels.append(int(edge[2]['label'][0]))
                    if len(edge[2]['label']) > 1:
                        edge_attributes.append(edge[2]['label'][1:])
            edge_slices.append(len(graph.edges()))
        # convert the edge indices to a tensor
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).T
        edge_slices = torch.tensor(edge_slices, dtype=torch.long).cumsum(dim=0)
        # convert the edge labels to a tensor
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        # apply row-wise one-hot encoding
        edge_labels = torch.nn.functional.one_hot(edge_labels).float()
        # convert the edge attributes to a tensor
        edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)
        if len(edge_attributes) == 0:
            edge_attributes = None
        if edge_attributes is not None:
            # stack edge attributes and edge labels together to form the edge feature matrix
            edge_attr = torch.cat((edge_attributes, edge_labels), dim=1)
        else:
            edge_attr = edge_labels
        y = torch.tensor(labels, dtype=torch.long)
        y_slices = torch.arange(0, len(labels)+1, dtype=torch.long)
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attr, y=y)
        slices = {'edge_index': edge_slices,
                    'x': node_slices,
                  'edge_attr': edge_slices.detach().clone(),
                  'y': y_slices}
        sizes = {'num_node_labels': node_labels.shape[1],
                 'num_node_attributes': node_attributes.shape[1] if node_attributes is not None else 0,
                 'num_edge_labels': edge_labels.shape[1],
                 'num_edge_attributes': edge_attributes.shape[1] if edge_attributes is not None else 0}
        return data, slices, sizes

    def read_nel_data_v2(self):
        load_path = Path(self.raw_dir)
        # load the nodes from the file
        node_labels = []
        node_attributes = None
        node_slices = [0]
        node_counter = 0
        unique_labels = []
        num_graphs = 0
        with open(load_path.joinpath(self.name + "_Nodes.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            # convert into torch tensor
            torch_lines = torch.zeros((len(lines), line_length), dtype=torch.float)
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(f'Processing node {i+1}/{len(lines)} in dataset {self.name}')
                line_data = line.strip().split(" ")
                torch_lines[i] = torch.tensor(list(map(float, line_data)))
            graph_ids = torch_lines[:, 0].long()
            # get slice vector from unique graph ids
            node_slices = torch.unique(graph_ids, return_counts=True)[1]
            num_graphs = len(node_slices)
            # add 0 at the beginning
            node_slices = torch.cat((torch.tensor([0]), node_slices)).cumsum(dim=0).long()
            node_ids = torch_lines[:, 1].long()
            node_labels = torch_lines[:, 2].long()
            unique_labels = len(torch.unique(node_labels))
            node_attr = None
            # sort the node labels graph-wise (node_slices) according to the node ids
            for idx in range(len(node_slices) - 1):
                sorted_indices = torch.argsort(node_ids[node_slices[idx]:node_slices[idx+1]])
                node_labels[node_slices[idx]:node_slices[idx+1]] = node_labels[node_slices[idx]:node_slices[idx+1]][sorted_indices]
                if line_length > 3:
                    node_attr = torch_lines[:, 3:]
                    node_attr[node_slices[idx]:node_slices[idx+1]] = node_attr[node_slices[idx]:node_slices[idx+1]][sorted_indices]

        x = None
        # one hot encoding if number of node labels is smaller than 100
        if unique_labels < 100:
            x = torch.nn.functional.one_hot(node_labels).float()
        else:
            x = node_labels
        if node_attr is not None:
            x = torch.cat((node_attr, x), dim=1)

        edge_indices = None
        edge_slices = None
        edge_labels = None
        edge_attr = None
        with open(load_path.joinpath(self.name + "_Edges.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            torch_lines = torch.zeros((len(lines), line_length), dtype=torch.float)
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(f'Processing edge {i+1}/{len(lines)} in dataset {self.name}')
                data = line.strip().split(" ")
                torch_lines[i] = torch.tensor(list(map(float, data)))
            graph_ids = torch_lines[:, 0].long()
            all_ids = torch.unique(graph_ids)
            # missing ids are those not in all_ids but in range(0, num_graphs)
            missing_ids = [i for i in range(num_graphs) if i not in all_ids]
            # get slice vector from unique graph ids
            edge_slices = torch.unique(graph_ids, return_counts=True)[1]
            # add 0 at the beginning
            edge_slices = torch.cat((torch.tensor([0]), edge_slices)).cumsum(dim=0).long()
            edge_slices = [x.item() for x in edge_slices]
            # duplicate the value at the index of the missing ids
            for missing_id in missing_ids:
                edge_slices.insert(missing_id+1, edge_slices[missing_id])
            edge_slices = torch.tensor(edge_slices, dtype=torch.long)
            edge_indices = torch_lines[:, 1:3].long().T
            edge_labels = torch_lines[:, 3].long()
            edge_labels = torch.nn.functional.one_hot(edge_labels).float()
            edge_attr = None
            if line_length > 4:
                edge_attr = torch_lines[:, 4:]
                edge_data = torch.cat((edge_attr, edge_labels), dim=1)
            else:
                edge_data = edge_labels

        y = None
        with open(load_path.joinpath(self.name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            torch_lines = torch.zeros((len(lines), line_length - 1), dtype=torch.long)
            for i, line in enumerate(lines):
                data = line.strip().split(" ")
                graph_name = data[0]
                torch_lines[i] = torch.tensor(list(map(float, data[1:])))
            y = torch_lines[:, 1].long()


        y_slices = torch.arange(0, len(y) + 1, dtype=torch.long)
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_data, y=y)
        slices = {'edge_index': edge_slices,
                  'x': node_slices,
                  'edge_attr': edge_slices.detach().clone(),
                  'y': y_slices}

        sizes = {'num_node_labels': unique_labels,
                 'num_node_attributes': node_attributes.shape[1] if node_attributes is not None else 0,
                 'num_edge_labels': edge_labels.shape[1],
                 'num_edge_attributes': edge_attr.shape[1] if edge_attr is not None else 0}
        return data, slices, sizes

    def create_nx_graph(self, graph_id: int, directed: bool = False):
        graph = self[graph_id]
        if isinstance(self.node_labels['primary'], NodeLabels):
            primary_labels = self.node_labels['primary'].node_labels[self.slices['x'][graph_id]:self.slices['x'][graph_id+1]]
        elif isinstance(self.node_labels['primary'], torch.Tensor):
            primary_labels = self.node_labels['primary'][self.slices['x'][graph_id]:self.slices['x'][graph_id+1]]
        else:
            raise ValueError('Node labels are not of type NodeLabels or torch.Tensor')
        nx_graph = to_networkx(
            data=graph,
            node_attrs=['x'],
            edge_attrs=['edge_attr'] if graph.edge_attr is not None else None,
            to_undirected=not directed)
        # change node label 'x' to 'primary_label'
        for node in nx_graph.nodes(data=True):
            nx_graph.nodes[node[0]]['primary_label'] = primary_labels[node[0]].item()
            del nx_graph.nodes[node[0]]['x']
        return nx_graph

    def create_nx_graphs(self, directed: bool = False):
        self.nx_graphs = []
        counter = 0
        for g_id, graph in enumerate(self):
            if g_id % 1000 == 0 or g_id == len(self) - 1:
                print(f'Processing graph {g_id+1}/{len(self)}')

            self.nx_graphs.append(to_networkx(
                data=graph,
                node_attrs=['x'] if self.task != 'node_classification' else None,
                edge_attrs=['edge_attr'] if graph.edge_attr is not None else None,
                to_undirected=not directed))

            # change node label 'x' to 'primary_label'
            for node in self.nx_graphs[-1].nodes(data=True):
                self.nx_graphs[-1].nodes[node[0]]['primary_label'] = self.node_labels['primary'][counter].item()
                if self.task != 'node_classification':
                    del self.nx_graphs[-1].nodes[node[0]]['x']
                counter += 1
            if graph.edge_attr is not None:
                unique_edge_labels = torch.unique(self.data.edge_attr)
                for edge in self.nx_graphs[-1].edges(data=True):
                    edge_label_one_hot = np.array(edge[2]['edge_attr'])[self.num_edge_attributes:]
                    if edge_label_one_hot.size == 1 :
                        # get one hot vector from edge_label_one_hot value
                        edge[2]['label'] = edge_label_one_hot
                    else:
                        edge[2]['label'] = np.argmax(edge_label_one_hot)
        pass

    def preprocess_share_gnn_data(self, data, input_features=None, output_features=None, task=None) -> None:
        if input_features is not None and task is not None:
            use_labels = input_features.get('name', 'node_labels') == 'node_labels'
            use_constant = input_features.get('name', 'node_labels') == 'constant'
            use_features = input_features.get('name', 'node_labels') == 'node_features'
            use_labels_and_features = input_features.get('name', 'node_labels') == 'all'
            transformation = input_features.get('transformation', None)
            use_features_as_channels = input_features.get('features_as_channels', False)

            use_train_node_labels = (task == 'node_classification') and input_features.get('one_hot_train_labels', False)

            ### Determine the input data
            if use_labels:
                data['x'] = data['x'][:, self.num_node_attributes:]
                if transformation in ['one_hot', 'one_hot_encoding']:
                    if data['x'].shape[1] == 1:
                        # convert to long tensor
                        data['x'] = data['x'].long()
                        data['x'] = torch.nn.functional.one_hot(data['x'].squeeze(1)).type(self.precision)
                else:
                    data['x'] = torch.argmax(data['x'], dim=1).unsqueeze(1)
            elif use_constant:
                data['x'] = torch.full(size=(data['x'].shape[0], input_features.get('in_dimensions', 1)), fill_value=input_features.get('value', 1.0), dtype=self.precision)
            elif use_features:
                data['x'] = data['x'][:, :self.num_node_attributes]
                if use_train_node_labels:
                    # get data y one hot
                    y_one_hot = torch.nn.functional.one_hot(data['y']).type(self.precision)
                    # set all rows of y_one_hot to 1/row_num if row is not in train mask
                    non_train_indices = (data['train_mask'] == 0).nonzero().squeeze()
                    y_one_hot[non_train_indices] = 1.0 / y_one_hot.shape[1]
                    data['x'] = torch.cat((y_one_hot, data['x']), dim=1)
            elif use_labels_and_features:
                # get first self.num_node_attributes columns and on the rest apply argmax
                data['x'] = torch.cat((data['x'][:, :self.num_node_attributes], torch.argmax(data['x'][:,self.num_node_attributes:], dim=1).unsqueeze(0)), dim=1)
            else:
                pass

            # normalize the graph input labels, i.e. to have values between -1 and 1, no zero values
            if use_labels and transformation == 'normalize':
                # get the number of unique node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # if value is even, add 1 to make it odd
                    if value % 2 == 0:
                        value = ((value + 1) * interval_length)
                    else:
                        value = (-1) * (value * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values
                data['x'] = data['x'].apply_(lambda x: normalized_node_labels[x])
            elif use_labels and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # map the value to the interval [0,1]
                    value = ((value + 1) * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values
                data['x'] = data['x'].apply_(lambda x: normalized_node_labels[x])
            elif use_labels and transformation == 'unit_circle':
                '''
                Arrange the labels in an 2D unit circle
                '''
                num_node_labels = self.unique_node_labels
                # duplicate data column
                data['x'] = data['x'].repeat(1, 2)
                data['x'] = data['x'][:, 0:1].apply_(lambda x: torch.cos(2 * np.pi * x / num_node_labels))
                data['x'] = data['x'][:, 1:2].apply_(lambda x: torch.sin(2 * np.pi * x / num_node_labels))
            elif use_labels_and_features and transformation == 'normalize_labels':
                # get the number of unique node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # if value is even, add 1 to make it odd
                    if value % 2 == 0:
                        value = ((value + 1) * interval_length)
                    else:
                        value = (-1) * (value * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values only for the last column
                data['x'] = data['x'][:, -1].apply_(lambda x: normalized_node_labels[x])
            elif use_labels_and_features and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # map the value to the interval [0,1]
                    value = ((value + 1) * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values only for the last column
                data['x'] = data['x'][:, -1].apply_(lambda x: normalized_node_labels[x])


            # Determine the output data
            #if task == 'regression':
            #    self.num_classes = 1
            #    if type(self.graph_labels[0]) == list:
            #        self.num_classes = len(self.graph_labels[0])
            #else:
            #    try:
            #        self.num_classes = len(set(self.graph_labels))
            #    except:
            #        self.num_classes = len(self.graph_labels[0])
            #
            # one hot encode y



            if task == 'graph_regression':
                if isinstance(output_features, dict):
                    if output_features.get('transformation', None) is not None:
                        data['y'] = transform_data(data['y'], output_features)
            elif task == 'node_classification':
                pass
                #data['y'] = torch.nn.functional.one_hot(data['y'], num_classes=self.num_classes).float()
            return None

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def relabel_most_frequent(labels: NodeLabels, num_max_labels: int):
    if num_max_labels is None:
        num_max_labels = -1
    # get the k most frequent node labels or relabel all
    if num_max_labels == -1:
        bound = len(labels.db_unique_node_labels)
    else:
        bound = min(num_max_labels, len(labels.db_unique_node_labels))
    most_frequent = sorted(labels.db_unique_node_labels, key=labels.db_unique_node_labels.get, reverse=True)[
                    :bound - 1]
    # relabel the node labels
    for i, _lab in enumerate(labels.node_labels):
        for j, lab in enumerate(_lab):
            if lab not in most_frequent:
                labels.node_labels[i][j] = bound - 1
            else:
                labels.node_labels[i][j] = most_frequent.index(lab)
    # set the new unique labels
    labels.num_unique_node_labels = bound
    db_unique = {}
    for i, l in enumerate(labels.node_labels):
        unique = {}
        for label in l:
            if label not in unique:
                unique[label] = 1
            else:
                unique[label] += 1
            if label not in db_unique:
                db_unique[label] = 1
            else:
                db_unique[label] += 1
        labels.unique_node_labels[i] = unique
    labels.db_unique_node_labels = db_unique
    pass


def transform_data(data, transformation_dict: Dict[str, Dict[str, str]]):
    # reformat the output data, shift to positive values and make the values smaller
    transformation_args = [{}]
    if transformation_dict.get('transformation_args', None) is not None:
        transformation_args = transformation_dict['transformation_args']
    if type(transformation_dict['transformation']) == list:
        for i, expression in enumerate(transformation_dict['transformation']):
            data = eval(expression)(input=data, **transformation_args[i])
    else:
        data = eval(transformation_dict['transformation'])(input=data, **transformation_args)
    return data

class GraphCount(InMemoryDataset):

    task_index = dict(
        triangle=0,
        tri_tail=1,
        star=2,
        cycle4=3,
        cycle5=4,
        cycle6=5,
        multi = -1,
    )

    def __init__(self, root:str, split:str, task:str, **kwargs):
        super().__init__(root=root, **kwargs)

        _pt = dict(zip(["train", "val", "test"], self.processed_paths))
        self.data, self.slices = torch.load(_pt[split])

        index = self.task_index[task]
        if index != -1:
            self.data.y = self.data.y[:, index:index+1]

    @property
    def raw_file_names(self):
        return ["Data/GraphDatasets/SubstructureCountingBenchmark.pt"]

    @property
    def processed_dir(self):
        return f"{self.root}/randomgraph"

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):

        _pt, = self.raw_file_names
        raw = torch.load(f"{self.root}/{_pt}")

        def to(graph):

            A = graph["A"]
            y = graph["y"]

            return pyg.data.Data(
                x=torch.ones(A.shape[0], 1, dtype=torch.int64), y=y,
                edge_index=torch.Tensor(np.vstack(np.where(graph["A"] > 0)))
                     .type(torch.int64),
            )

        data = [to(graph) for graph in raw["data"]]

        if self.pre_filter is not None:
            data = filter(self.pre_filter, data)

        if self.pre_transform is not None:
            data = map(self.pre_transform, data)

        data_list = list(data)
        normalize = torch.std(torch.stack([data.y for data in data_list]), dim=0)

        for split in ["train", "val", "test"]:

            from operator import itemgetter
            split_idx = raw["index"][split]
            splits = itemgetter(*split_idx)(data_list)

            data, slices = self.collate(splits)
            data.y = data.y / normalize

            torch.save((data, slices), f"{self.processed_dir}/{split}.pt")
