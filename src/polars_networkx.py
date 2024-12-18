import polars as pl
from polars import col as c
from typing import Optional
import networkx as nx

from general_function import modify_string, generate_log, generate_uuid

from config import settings

# Global variable
log = generate_log(name=__name__)


def generate_nx_edge(data: pl.Expr, nx_graph: nx.Graph) -> pl.Expr:
    return data.map_elements(lambda x: nx_graph.add_edge(**x), return_dtype=pl.Struct)


def get_edge_data_list(nx_graph: nx.Graph, data_name: str) -> list:
    return list(map(lambda x: x[-1][data_name], nx_graph.edges(data=True)))

