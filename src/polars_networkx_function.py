import polars as pl
from polars import col as c
from typing import Optional
import networkx as nx

from general_function import generate_log


# Global variable
log = generate_log(name=__name__)


def generate_nx_edge(data: pl.Expr, nx_graph: nx.Graph) -> pl.Expr:
    """
    Generate edges in a NetworkX graph from a Polars expression.

    Args:
        data (pl.Expr): The Polars expression containing edge data.
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        pl.Expr: The Polars expression with edges added to the graph.
    """
    return data.map_elements(lambda x: nx_graph.add_edge(**x), return_dtype=pl.Struct)


def get_edge_data_list(nx_graph: nx.Graph, data_name: str) -> list:
    """
    Get a list of edge data from a NetworkX graph.

    Args:
        nx_graph (nx.Graph): The NetworkX graph.
        data_name (str): The name of the edge data to retrieve.

    Returns:
        list: The list of edge data.
    """
    return list(map(lambda x: x[-1][data_name], nx_graph.edges(data=True)))

def get_edge_data_from_node_list(node_list: list, nx_graph: nx.Graph) -> list[dict]:
    """
    Get edge data for a list of nodes from a NetworkX graph.

    Args:
        node_list (list): The list of node IDs.
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        list[dict]: The list of dictionaries containing every edge data.
    """
    return list(map(
        lambda x: {"u_of_edge": x[0],"v_of_edge": x[1]} | x[2], 
        nx.subgraph(nx_graph, node_list).edges(data=True)
    ))


def get_shortest_path(node_id_list: list, nx_graph: nx.Graph, weight: str="length") -> list[str]:
    """
    Get the shortest path between two nodes in a NetworkX graph.

    Args:
        node_id_list (list): The list of node IDs.
        nx_graph (nx.Graph): The NetworkX graph.
        weight (str, optional): The edge weight attribute. Defaults to "length".

    Returns:
        list[str]: The list of node IDs in the shortest path.
    """
    return list(nx.shortest_path(nx_graph, source=node_id_list[0], target=node_id_list[-1], weight=weight))


def highlight_connected_edges(nx_graph: nx.Graph) -> pl.DataFrame:
    """
    Highlight connected edges in a NetworkX graph and return them as a Polars DataFrame.

    Args:
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the connected edges with columns 'graph_id', 'u_of_edge', 
        'v_of_edge', and every edge attribute.
    """

    edge_data = list(map(
        lambda x: get_edge_data_from_node_list(node_list=x, nx_graph=nx_graph), 
        nx.connected_components(nx_graph)
    ))
    return pl.DataFrame(
        zip(range(len(edge_data)), edge_data),
        schema=["graph_id", "data"]
    ).explode("data").unnest("data")