import networkx as nx
def plot_graph(nx_graph: nx.Graph, with_node_ids: bool = False):
    """
    Plot the given networkx graph.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(nx_graph)
    nx.draw_networkx_nodes(nx_graph, pos, node_size=700, node_color='lightblue')
    node_labels = nx.get_node_attributes(nx_graph, 'primary_label')
    node_ids = [node_id for node_id in nx_graph.nodes()] if with_node_ids else None
    if node_ids is not None:
        node_labels = {node_id: f"{node_labels[node_id]} ({node_id})" for node_id in node_ids if node_id in node_labels}
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12)
    nx.draw_networkx_edges(nx_graph, pos)
    plt.show()

def plot_graph_changes(nx_graph: nx.Graph, edge=None, node=None, type=None, title=None):
    """
    Plot the given networkx graph.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(nx_graph)
    node_colors = ['lightblue'] * nx_graph.number_of_nodes()
    # if and edge is removed, color the endpoint nodes red
    if edge is not None and (type == 'remove' or type == 'add'):
            node_colors[edge[0]] = 'red'
            node_colors[edge[1]] = 'red'
    if node is not None and (type == 'remove' or type == 'add'):
        node_colors[node] = 'red'
    nx.draw_networkx_nodes(nx_graph, pos, node_size=700, node_color=node_colors)
    node_labels = nx.get_node_attributes(nx_graph, 'primary_label')
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12)
    nx.draw_networkx_edges(nx_graph, pos)
    plt.title(title if title is not None else '')
    plt.show()