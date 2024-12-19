import polars as pl
from polars import col as c
from typing import Optional
import networkx as nx

from general_function import modify_string, generate_log, generate_uuid

from config import settings

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