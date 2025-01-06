import unittest
import polars as pl
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from networkx_function import (
    generate_nx_edge, get_edge_data_list, get_edge_data_from_node_list,
    get_shortest_path, get_connected_edges_data, generate_and_connect_segment_from_linestring_list
)

class TestNetworkxFunctions(unittest.TestCase):

    def setUp(self):
        self.nx_graph = nx.Graph()
        self.nx_graph.add_edge("A", "B", length=1)
        self.nx_graph.add_edge("B", "C", length=2)
        self.nx_graph.add_edge("C", "D", length=1)
        self.nx_graph.add_edge("D", "E", length=2)
        self.nx_graph.add_edge("E", "F", length=1)

    def test_generate_nx_edge(self):
        df = pl.DataFrame({"u_of_edge": ["X"], "v_of_edge": ["Y"], "length": [3]})
        result = df.with_columns(generate_nx_edge(pl.struct(["u_of_edge", "v_of_edge", "length"]), self.nx_graph))
        self.assertIn(("X", "Y"), self.nx_graph.edges)

    def test_get_edge_data_list(self):
        result = get_edge_data_list(self.nx_graph, "length")
        self.assertEqual(result, [1, 2, 1, 2, 1])

    def test_get_edge_data_from_node_list(self):
        node_list = ["A", "B", "C"]
        result = get_edge_data_from_node_list(node_list, self.nx_graph)
        expected = [
            {"u_of_edge": "A", "v_of_edge": "B", "length": 1},
            {"u_of_edge": "B", "v_of_edge": "C", "length": 2}
        ]
        self.assertEqual(result, expected)

    def test_get_shortest_path(self):
        node_id_list = ["A", "F"]
        result = get_shortest_path(node_id_list, self.nx_graph)
        self.assertEqual(result, ["A", "B", "C", "D", "E", "F"])

    def test_get_connected_edges_data(self):
        result = get_connected_edges_data(self.nx_graph)
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.columns, ["graph_id", "u_of_edge", "v_of_edge", "length"])

    def test_generate_and_connect_segment_from_linestring_list(self):
        linestring_list = [
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(3, 3), (4, 4), (5, 5)]),
            LineString([(6, 6), (7, 7), (8, 8)]),
            LineString([(9, 9), (10, 10), (11, 11)]),
            LineString([(12, 12), (13, 13), (14, 14)]),
            LineString([(15, 15), (16, 16), (17, 17)]),
            LineString([(18, 18), (19, 19), (20, 20)]),
            LineString([(21, 21), (22, 22), (23, 23)]),
            LineString([(24, 24), (25, 25), (26, 26)]),
            LineString([(27, 27), (28, 28), (29, 29)])
        ]
        result = generate_and_connect_segment_from_linestring_list(linestring_list)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(ls, LineString) for ls in result))

if __name__ == "__main__":
    unittest.main()
