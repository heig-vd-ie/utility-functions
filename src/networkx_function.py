import polars as pl
from polars import col as c
from typing import Optional, Union
import networkx as nx
import graphblas as gb
from shapely.geometry import LineString, MultiLineString, MultiPoint
from shapely.ops import nearest_points

from shapely_function import (
    segment_list_from_multilinestring, shape_list_to_wkt_list, multipoint_from_multilinestring)
from polars_shapely_function import (
    get_linestring_boundaries_col, get_multigeometry_from_col, shape_intersect_shape_col)

from general_function import generate_log


# Global variable
log = generate_log(name=__name__)


def get_shortest_path_edge_data(nx_graph: nx.Graph, source, target, data_name: str, weight: str ='weight'):
    """
    Get the edge data along the shortest path between source and target nodes in a graph.

    Args:
        nx_graph (nx.Graph): The graph to search.
        source (node): Starting node for the path.
        target (node): Ending node for the path.
        data_name (str): The name of the edge attribute to retrieve.
        weight (str, optional): The edge attribute to use as weight. Default is 'weight'.
    Returns:
        list: A list of edge data along the shortest path.
    """
    path = nx.shortest_path(nx_graph, source=source, target=target, weight=weight)
    if path:
        return list(map(lambda x: x[-1][data_name], list(nx.subgraph(nx_graph, path).edges(data=True))))
    else:
        return None

def get_node_neighbor_edge_data(nx_graph: nx.Graph, node):
    """
    Get the edge data for all neighbors of a specified node in a graph.

    Args:
        nx_graph (nx.Graph): The graph to search.
        node (node): The node whose neighbors' edge data is to be retrieved.

    Returns:
        list: A list of all neighbors edge data of the specified node.
    """
    return list(map(
        lambda x: {"u_of_edge": x[0],"v_of_edge": x[1]} | x[2], 
        nx_graph.edges(node, data=True))
    )

def get_shortest_path_dijkstra_from_multisource(nx_graph: nx.Graph, source: list, target, weight: str ='weight'):
    """
    Get the shortest path between source and target nodes using Dijkstra's algorithm.

    Args:
        nx_graph (nx.Graph): The graph to search.
        source (list): Starting node for the path.
        target: Ending node for the path.
        weight (str, optional): The edge attribute to use as weight. Default is 'weight'.

    Returns:
    list: A list of nodes in the shortest path.
    """
    return nx.multi_source_dijkstra(nx_graph, sources=set(source).difference(list(target)), target=target, weight=weight)[1]

def get_shortest_path_dijkstra_col_from_multisource(
    target: pl.Expr, nx_graph: nx.Graph, source: list, weight='weight') -> pl.Expr:
    """
    Get the shortest path between source and target nodes using Dijkstra's algorithm. Targets are stored in a polars columns  
    and return it as a column.

    Args:
        nx_graph (nx.Graph): The graph to search.
        source (list): Starting node for the path.
        target (pl.Expr): Ending node for the path.
        weight (str, optional): The edge attribute to use as weight. Default is 'weight'.

    Returns:
        pl.Expr: The Polars expression containing lists of shortest path nodes.
    """
    return (
        target.map_elements(lambda x:
            get_shortest_path_dijkstra_from_multisource(target=x, nx_graph=nx_graph, source=source, weight=weight),
            return_dtype=pl.List(pl.Utf8))
    )
    


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

def get_all_edge_data(nx_graph: nx.Graph) -> pl.DataFrame:
    """
    Get every edge data from a NetworkX graph.

    Args:
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        pl.DataFrame: Polar DataFrame containing every edge data.
    """
    return pl.DataFrame(
        list(nx_graph.edges(data=True)), 
        strict=False, orient="row", 
        schema=["u_of_edge", "v_of_edge", "data"]
    ).unnest("data")

def get_edge_param_from_node_list(node_list: list, nx_graph: nx.Graph, data_name: str) -> list[dict]:
    """
    Get edge parameter from a list of nodes from a NetworkX graph.

    Args:
        node_list (list): The list of node IDs.
        nx_graph (nx.Graph): The NetworkX graph.
        data_name(str): The name of the edge parameter to retrieve.
    Returns:
        list[dict]: The list of dictionaries containing every edge data.
    """
    return list(map(
        lambda x:  x[-1][data_name], nx.subgraph(nx_graph, node_list).edges(data=True)
    ))


def get_connected_edges_data(nx_graph: nx.Graph) -> pl.DataFrame:
    """
    Group all edges that are connected together in a NetworkX graph and return the result as a Polars DataFrame.

    Args:
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the connected edges with columns `graph_id` , `u_of_edge`, 
        `v_of_edge`, and every edge attribute.
    """

    edge_data = list(map(
        lambda x: get_edge_data_from_node_list(node_list=x, nx_graph=nx_graph), 
        nx.connected_components(nx_graph)
    ))
    return pl.DataFrame({"data": edge_data}, strict=False).with_row_index("graph_id").explode("data").unnest("data")


def generate_and_connect_segment_from_linestring_list(linestring_list: list[LineString]) -> list[LineString]:
    """
    Generate unitary segments from a list of LineString objects and ensure every segment is
    connected in a NetworkX graph sense:

    1.  Break down the LineString objects: Divide each LineString into its individual segments, each defined by two 
        consecutive points (start and end).
    2.  Build a graph: Create a NetworkX graph where each segment is an edge, and the endpoints of the segments serve 
        as nodes.
    3.  Define every connected subgraph.
    4.  Connect every subgraph finding the smallest segments needed.
    
    Args:
        linestring_list (list[LineString]): The list of LineString objects.

    Returns:
        list[LineString]: The list of connected LineString segments.
    """

    segment_list: list[LineString] = segment_list_from_multilinestring(MultiLineString(linestring_list))

    segment_pl: pl.DataFrame  = pl.DataFrame({
        "geometry" : shape_list_to_wkt_list(segment_list) # type: ignore
        }).with_columns(
            c("geometry").pipe(get_linestring_boundaries_col).alias("node_id"),
            c("geometry").pipe(get_linestring_boundaries_col).alias("edge_id")
            .list.to_struct(fields=["v_of_edge", "u_of_edge"])
        ).unnest("edge_id")
        
    nx_graph = nx.Graph()
    _ = segment_pl.with_columns(
            pl.struct("v_of_edge", "u_of_edge", "geometry").pipe(generate_nx_edge, nx_graph= nx_graph)
        )  

    if nx.is_connected(nx_graph):
        return segment_list  

    connected_edge: pl.DataFrame = get_connected_edges_data(nx_graph=nx_graph)

    graph_connected: list[int] = []
    for graph_id in connected_edge["graph_id"].unique():
        if graph_id not in graph_connected:
            if graph_connected:
                graph_id_to_check: list[int] = graph_connected
            else:
                graph_id_to_check: list[int] = connected_edge.filter(c("graph_id") != graph_id)["graph_id"].unique().to_list()
                
            point_to_connect: MultiPoint = multipoint_from_multilinestring(
                get_multigeometry_from_col(connected_edge.filter(c("graph_id") == graph_id))) # type: ignore
            point_to_check: MultiPoint = multipoint_from_multilinestring(
                get_multigeometry_from_col(connected_edge.filter(c("graph_id").is_in(graph_id_to_check))) # type: ignore
            )

            new_segment_points = nearest_points(point_to_connect, point_to_check)
                
            graph_connected.extend(
                connected_edge
                .filter(c("geometry").pipe(shape_intersect_shape_col, geometry=MultiPoint(new_segment_points)))
                ["graph_id"].unique().to_list()
            )
            segment_list.append(LineString(new_segment_points))
    return segment_list

def generate_bfs_tree_with_edge_data(graph: nx.Graph, source):
    """
    Create a BFS tree from a graph while retaining edge data.
    
    Parameters:
        graph (nx.Graph): The input graph.
        source (node): The starting node for BFS.
    
    Returns:
        nx.DiGraph: A directed BFS tree with edge data preserved.
    """
    # Create an empty directed graph for the BFS tree
    bfs_tree = nx.DiGraph()
    
    # Add nodes to the BFS tree
    
    # Add edges to the BFS tree with preserved edge data
    for u, v in nx.bfs_edges(graph, source):
        
        edge_data = graph.get_edge_data(u, v)
        bfs_tree.add_edge(u, v, **edge_data)
    
    return bfs_tree

def generate_shortest_path_length_matrix(
    nx_grid: nx.Graph, weight_name: Optional[str] = None, forced_weight: Optional[Union[int, float]] = None
    ) -> gb.Matrix: # type: ignore
    """
    Generate a matrix of shortest path lengths between all pairs of nodes in a graph.

    Args:
        nx_graph (nx.Graph): The graph to process.
        weight (str, optional): The edge attribute to use as weight. Default is 'weight'.
        forced_weight (Optional[Union[int, float]], optional): The weight to use for all edges. Default is None.
        
    Returns:
        gb.Matrix: A GraphBlas matrix where the rows and columns represent the nodes (from and to), 
        and the values represent the shortest path lengths.

    """
    # ...existing code...
    shortest_path = list(zip(*nx.shortest_path_length(nx_grid, weight=weight_name)))
    h_pl: pl.DataFrame = pl.DataFrame({
        "x": shortest_path[0],
        "data": list(map(lambda x: list(x.items()), list(shortest_path[1])))
    }, strict=False).explode("data").with_columns(
        c("data").list.to_struct(fields=["y", "weight"]).alias("data")
    ).unnest("data").with_columns(
        pl.lit(1).alias("weight")
    )
    value = [forced_weight]*h_pl.height if forced_weight is not None else h_pl["weight"].to_list()

    h_gb: gb.Matrix = gb.Matrix.from_coo( # type: ignore
        h_pl["x"].to_list(),
        h_pl["y"].to_list(),
        value,
        nrows=len(nx_grid), ncols=len(nx_grid)
    )
    return h_gb