import networkx as nx
import numpy as np
from utils.plotting import plot_graph, plot_graph_changes

class EditPath:
    """
    This class stores and reformats the output of the edit path function of networkx.
    """
    def __init__(self, db_name=None, start_id=None, end_id=None, start_graph:nx.Graph=None, end_graph:nx.Graph=None, edit_path=None, iteration=None, max_iterations=None, timeout=None):
        """
        Initialize the EditPath object with the edit path information.
        :param db_name: Name of the database or dataset this edit path belongs to.
        :param start_id: Id of the starting graph in the edit path, i.e., position in the dataset.
        :param end_id: Id of the ending graph in the edit path, i.e., position in the dataset.
        :param start_graph: NetworkX graph representing the starting graph of the edit path.
        :param end_graph: NetworkX graph representing the ending graph of the edit path.
        :param edit_path: Output of the function nx.optimize_edit_paths, which is a tuple containing:
            - A list of node operations, where each operation is a tuple (node1, node2) indicating the operation between two nodes.
            - A list of edge operations, where each operation is a tuple (edge1, edge2) indicating the operation between two edges.
        :param iteration: The current iteration of the edit path generation, useful for tracking progress.
        :param max_iterations: The maximum number of iterations to generate the edit path.
        :param timeout: The maximum time in seconds to generate the edit path.
        """

        if edit_path is not None:
            # node operations
            self.distance = edit_path[2]
            node_operations_all = [(a, b) for (a, b) in edit_path[0] if a != b]
            self.node_operations = dict()
            self.node_operations['remove'] = [a for (a, b) in node_operations_all if b is None]
            self.node_operations['add'] = [b for (a, b) in node_operations_all if a is None]
            if nx.get_node_attributes(start_graph, 'primary_label') and nx.get_node_attributes(end_graph, 'primary_label'):
                self.node_operations['relabel'] = [(a, b) for (a, b) in node_operations_all if a is not None and b is not None and start_graph.nodes[a]['primary_label'] != end_graph.nodes[b]['primary_label']]
            self.node_map = dict()
            self.inverse_node_map = dict()
            for (a, b) in node_operations_all:
                if a is not None and b is not None:
                    self.node_map[a] = b
                    self.inverse_node_map[b] = a
            # edge operations
            edge_operations_all = [(a, b) for (a, b) in edit_path[1] if a != b]
            self.edge_operations = dict()
            self.edge_operations['remove'] = [a for (a, b) in edge_operations_all if b is None]
            self.edge_operations['add'] = [b for (a, b) in edge_operations_all if a is None]
            # check whether 'label' is in the edge attributes
            if nx.get_edge_attributes(start_graph, 'label') and nx.get_edge_attributes(end_graph, 'label'):
                self.edge_operations['relabel'] = [(a, b) for (a, b) in edge_operations_all if a is not None and b is not None and start_graph.edges[a]['label'] != end_graph.edges[b]['label']]
            self.all_operations = list()
            for key, value in self.node_operations.items():
                self.all_operations.extend([(f'{key}_node', op) for op in value])
            for key, value in self.edge_operations.items():
                self.all_operations.extend([(f'{key}_edge', op) for op in value])


            self.db_name = db_name
            self.start_id = start_id
            self.end_id = end_id
            self.iteration = iteration
            self.max_iterations = max_iterations
            self.timeout = timeout

    # serialize the class to a json object
    def toJSON(self):
        return {
            'db_name': self.db_name,
            'start_id': self.start_id,
            'end_id': self.end_id,
            'distance': self.distance,
            'iteration': self.iteration,
            'max_iterations': self.max_iterations,
            'timeout': self.timeout,
            'node_operations': self.node_operations,
            'edge_operations': self.edge_operations,
            'all_operations': self.all_operations,
            'node_map': self.node_map,
            'inverse_node_map': self.inverse_node_map,
        }
    def loadJSON(self, json_obj):
        """
        Load the edit path from a JSON object.
        """
        self.db_name = json_obj['db_name']
        self.start_id = json_obj['start_id']
        self.end_id = json_obj['end_id']
        self.distance = json_obj['distance']
        self.node_operations = json_obj['node_operations']
        self.edge_operations = json_obj['edge_operations']
        self.iteration = json_obj['iteration']
        self.max_iterations = 0
        self.timeout = 0
        if 'max_iterations' in json_obj:
            self.max_iterations = json_obj['max_iterations']
        if 'timeout' in json_obj:
            self.timeout = json_obj['timeout']
        if 'all_operations' in json_obj:
            self.all_operations = json_obj['all_operations']
        else:
            self.all_operations = list()
            for key, value in self.node_operations.items():
                self.all_operations.extend([(f'{key}_node', op) for op in value])
            for key, value in self.edge_operations.items():
                self.all_operations.extend([(f'{key}_edge', op) for op in value])
        if 'node_map' in json_obj:
            self.node_map = json_obj['node_map']
        if 'inverse_node_map' in json_obj:
            self.inverse_node_map = json_obj['inverse_node_map']
        else:
            self.inverse_node_map = dict()
            for key, value in self.node_map.items():
                self.inverse_node_map[value] = key
        return self

    def apply_operations(self, operations_list, graph_sequence, target_graph, plotting=False):
        unsuccessful_operations = []
        for op_type, op_value in operations_list:
            # create a copy of the last graph in the sequence
            last_graph = graph_sequence[-1].copy()
            # differentiate between operation types
            if op_type == 'add_node':
                # add a node with the given value
                last_graph.add_node(op_value, primary_label=op_value)
                graph_sequence.append(last_graph)
                if plotting:
                    plot_graph_changes(graph_sequence[-1], node=op_value, type='add', title='Add Node')
            elif op_type == 'remove_node':
                # remove the node with the given value (only if it degree is 0)
                if last_graph.has_node(op_value) and last_graph.degree(op_value) == 0:
                    last_graph.remove_node(op_value)
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], node=op_value, type='remove', title='Remove Node')
                else:
                    unsuccessful_operations.append((op_type, op_value))
            elif op_type == 'relabel_node':
                # relabel the node with the given value
                if last_graph.has_node(op_value[0]):
                    last_graph.nodes[op_value[0]]['primary_label'] = target_graph.nodes[op_value[1]]['primary_label']
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], node=op_value[0], type='add', title='Relabel Node')
            elif op_type == 'add_edge':
                # add an edge between the two nodes with the given values
                # first use inverse_node_map to get the correct node ids
                head_node = op_value[0]
                tail_node = op_value[1]
                if op_value[0] in self.inverse_node_map:
                    head_node = self.inverse_node_map[op_value[0]]
                if op_value[1] in self.inverse_node_map:
                    tail_node = self.inverse_node_map[op_value[1]]

                if last_graph.has_node(head_node) and last_graph.has_node(tail_node):
                    last_graph.add_edge(head_node, tail_node)
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], edge=op_value, type='add', title='Add Edge')
                    pass
            elif op_type == 'remove_edge':
                # first use inverse_node_map to get the correct node ids
                head_node = op_value[0]
                tail_node = op_value[1]
                # remove the edge between the two nodes with the given values
                if last_graph.has_edge(head_node, tail_node):
                    last_graph.remove_edge(head_node, tail_node)
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], edge=op_value, type='remove', title='Remove Edge')
                    pass
            elif op_type == 'relabel_edge':
                if last_graph.has_edge(*op_value[0]):
                    last_graph.edges[op_value[0]]['label'] = target_graph.edges[op_value[1]]['label']
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], edge=op_value[0], type='add', title='Relabel Edge')
                    pass
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
        return unsuccessful_operations

    def create_edit_path_graphs(self, nx_graph1, nx_graph2, seed=42, plotting=True):
        """
        Create a sequence of networkx graphs representing the edit path. Starting from nx_graph1, applying the node and edge operations
        and ending with nx_graph2.
        """
        graph_sequence = [nx_graph1]
        if plotting:
            plot_graph(nx_graph1, with_node_ids=True)
            plot_graph(nx_graph2, with_node_ids=True)
        # create a shuffled list of operations to apply
        unsuccessful_operations = self.all_operations.copy()
        np.random.seed(seed)
        np.random.shuffle(unsuccessful_operations)
        while len(unsuccessful_operations) > 0:
            unsuccessful_operations = self.apply_operations(unsuccessful_operations, graph_sequence, target_graph=nx_graph2, plotting=plotting)

        # TODO: add the node operations to the graph1 and graph2
        plot_graph(nx_graph2)
        return graph_sequence
