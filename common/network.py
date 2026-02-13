import networkx as nx


def generate_network(node_list):
    """
    Build a NetworkX graph of stations.

    Args:
        node_list (list[int]): Number of stations per category.

    Returns:
        nx.Graph: Graph where each node has attributes:
            - "station": category label
            - "bikes": initial bike count (0)

    Example:
        >>> g = generate_network([2, 1])
        >>> g.number_of_nodes()
        3
        >>> g.nodes[0]
        {'station': 0, 'bikes': 0}
    """
    g = nx.Graph()

    n_categories = len(node_list)
    if n_categories == 2:
        categories = [0, 4]
    elif n_categories == 3:
        categories = [0, 2, 4]
    elif n_categories == 4:
        categories = [0, 1, 3, 4]
    else:
        categories = [0, 1, 2, 3, 4]

    prev = 0
    for i in range(n_categories):
        if i > 0:
            category_i_nodes = range(prev, prev + node_list[i])
            prev += node_list[i]
        else:
            category_i_nodes = range(node_list[i])
            prev = node_list[i]

        for node in category_i_nodes:
            g.add_node(node, station=categories[i], bikes=0)

    return g
