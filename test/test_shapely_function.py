import unittest
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon, LineString, MultiLineString
from shapely_function import (
    point_list_to_linestring, get_polygon_multipoint_intersection, find_closest_node_from_list,
    explode_multipolygon, geoalchemy2_to_shape, shape_to_geoalchemy2, get_closest_point_from_multi_point,
    remove_z_coordinates, get_valid_polygon_str, partition, generate_valid_polygon,
    shape_list_to_wkt_list, segment_list_from_multilinestring
)

class TestShapelyFunctions(unittest.TestCase):

    def test_point_list_to_linestring(self):
        point_list = ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"]
        result = point_list_to_linestring(point_list)
        self.assertEqual(result, "LINESTRING (0 0, 1 1, 2 2)")

    def test_get_polygon_multipoint_intersection(self):
        polygon_str = "POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"
        multipoint = MultiPoint([(1, 1), (3, 3)])
        result = get_polygon_multipoint_intersection(polygon_str, multipoint)
        self.assertEqual(result, ["POINT (1 1)"])

    def test_find_closest_node_from_list(self):
        data = {
            "node_id": "POINT (0 0)",
            "node_list_name": ["POINT (1 1)", "POINT (2 2)", "POINT (0.5 0.5)"]
        }
        result = find_closest_node_from_list(data, "node_id", "node_list_name")
        self.assertEqual(result, "POINT (0.5 0.5)")

    def test_explode_multipolygon(self):
        multipolygon_str = "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"
        result = explode_multipolygon(multipolygon_str)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Polygon)
        self.assertIsInstance(result[1], Polygon)

    def test_geoalchemy2_to_shape(self):
        geo_str = "0101000000000000000000F03F0000000000000040"  # WKB for POINT (1 2)
        result = geoalchemy2_to_shape(geo_str)
        self.assertIsInstance(result, Point)
        self.assertEqual(result, Point(1, 2))

    def test_shape_to_geoalchemy2(self):
        point = Point(1, 2)
        result = shape_to_geoalchemy2(point)
        self.assertTrue(result.startswith("0101000000"))

    def test_get_closest_point_from_multi_point(self):
        geo_str = "POINT (0 0)"
        multi_point = MultiPoint([(1, 1), (2, 2), (0.5, 0.5)])
        result = get_closest_point_from_multi_point(geo_str, multi_point)
        self.assertEqual(result, "POINT (0.5 0.5)")

    def test_remove_z_coordinates(self):
        point = Point(1, 2, 3)
        result = remove_z_coordinates(point)
        self.assertEqual(result.wkt, "POINT (1 2)") # type: ignore

    def test_partition(self):
        polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        result = partition(polygon, 1)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]))

    def test_generate_valid_polygon(self):
        multipolygon_str = "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"
        invalid_multipolygon_str = "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0, 0.5 0.5, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"
        value = MultiPolygon([Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])])
        result = generate_valid_polygon(multipolygon_str)
        self.assertIsInstance(result, MultiPolygon)
        self.assertEqual(result, value) # type: ignore
        result = generate_valid_polygon(invalid_multipolygon_str)
        self.assertIsInstance(result, MultiPolygon)
        self.assertEqual(result, value) # type: ignore
        linestring_str  = "LINESTRING (0 0, 0 1, 1 1, 1 0)"
        result = generate_valid_polygon(linestring_str)
        self.assertEqual(result, None) # type: ignore

    def test_shape_list_to_str_list(self):
        shape_list = [Point(1, 1), LineString([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
        result = shape_list_to_wkt_list(shape_list)
        expected = [
            "POINT (1 1)",
            "LINESTRING (0 0, 1 1)",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        ]
        self.assertEqual(result, expected)

    def test_segment_list_from_multilinestring(self):
        multi_linestring = MultiLineString([[(0, 0), (1, 1), (5, 3), (6, 3)], [(0, 10), (1, 1), (2, 2)]])
        result = segment_list_from_multilinestring(multi_linestring)
        expected = [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (5, 3)]),
            LineString([(5, 3), (6, 3)]),
            LineString([(0, 10), (1, 1)]),
            LineString([(1, 1), (2, 2)])
        ]
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
