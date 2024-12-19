import unittest
import polars as pl
from polars import col as c 
from shapely.geometry import Point, Polygon, MultiPoint, LineString
from shapely import set_precision
from polars_shapely_function import (
    shape_intersect_polygon, get_linestring_boundaries_col, get_geometry_list, get_multigeometry_from_col,
    add_buffer, calculate_line_length, shape_coordinate_transformer_col, generate_point_from_coordinates,
    generate_shape_linestring, get_linestring_from_point_list, combine_shape, shape_to_wkt_col, wkt_to_shape_col,
    geojson_to_wkt_col, shape_to_geoalchemy2_col, geoalchemy2_to_shape_col, wkt_to_geoalchemy_col, geoalchemy2_to_wkt_col
)

class TestPolarsShapelyFunctions(unittest.TestCase):

    def test_shape_intersect_polygon(self):
        df = pl.DataFrame({"geometry": ["POINT (1 1)", "POINT (3 3)"]})
        polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        result = df.with_columns(shape_intersect_polygon(pl.col("geometry"), polygon).alias("intersects"))
        self.assertEqual(result["intersects"].to_list(), [True, False])

    def test_get_linestring_boundaries_col(self):
        df = pl.DataFrame({"geometry": ["LINESTRING (0 0, 1 1, 2 2)"]})
        result = df.with_columns(get_linestring_boundaries_col(pl.col("geometry")).alias("boundaries"))
        self.assertEqual(result["boundaries"].to_list(), [["POINT (0 0)", "POINT (2 2)"]])

    def test_get_geometry_list(self):
        df = pl.DataFrame({"geometry": ["POINT (1 1)", "POINT (2 2)"]})
        result = get_geometry_list(df)
        self.assertEqual(result, [Point(1, 1), Point(2, 2)])

    def test_get_multigeometry_from_col(self):
        df = pl.DataFrame({"geometry": ["POINT (1 1)", "POINT (2 2)"]})
        result = get_multigeometry_from_col(df)
        self.assertIsInstance(result, MultiPoint)
        self.assertEqual(len(result.geoms), 2)

    def test_add_buffer(self):
        df = pl.DataFrame({"geometry": ["POINT (1 1)"]})
        result = df.with_columns(add_buffer(pl.col("geometry"), 1.0).alias("buffered"))
        self.assertTrue(result["buffered"][0].startswith("POLYGON"))

    def test_calculate_line_length(self):
        df = pl.DataFrame({"geometry": ["LINESTRING (0 0, 3 4)"]})
        result = df.with_columns(calculate_line_length(pl.col("geometry")).alias("length"))
        self.assertEqual(result["length"][0], 5.0)

    def test_shape_coordinate_transformer_col(self):
        df = pl.DataFrame({"geometry": [Point(1, 1)]})
        result = df.with_columns(c("geometry").pipe(
            shape_coordinate_transformer_col, crs_from= 4326, crs_to = 2056).alias("transformed"))
        self.assertEqual(set_precision(result["transformed"][0], 1), Point(1576839, -4479629))

    def test_generate_point_from_coordinates(self):
        df = pl.DataFrame({"x": [1.0], "y": [2.0]})
        result = df.with_columns(generate_point_from_coordinates(pl.col("x"), pl.col("y")).alias("point"))
        self.assertEqual(result["point"][0], "POINT (1 2)")

    def test_generate_shape_linestring(self):
        df = pl.DataFrame({"coords": [[0, 0, 1, 1, 2, 2]]})
        result = df.with_columns(generate_shape_linestring(pl.col("coords")).alias("linestring"))
        self.assertEqual(result["linestring"][0], "LINESTRING (0 0, 1 1, 2 2)")

    def test_get_linestring_from_point_list(self):
        df = pl.DataFrame({"points": [["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"]]})
        result = df.with_columns(get_linestring_from_point_list(pl.col("points")).alias("linestring"))
        self.assertEqual(result["linestring"][0], "LINESTRING (0 0, 1 1, 2 2)")

    def test_combine_shape(self):
        df = pl.DataFrame({"geometries": [["POINT (0 0)", "POINT (1 1)"]]})
        result = df.with_columns(combine_shape(pl.col("geometries")).alias("combined"))
        self.assertTrue(result["combined"][0].startswith("MULTIPOINT"))

    def test_shape_to_wkt_col(self):
        df = pl.DataFrame({"geometry": [Point(1, 1)]})
        result = df.with_columns(shape_to_wkt_col(pl.col("geometry")).alias("wkt"))
        self.assertEqual(result["wkt"][0], "POINT (1 1)")

    def test_wkt_to_shape_col(self):
        df = pl.DataFrame({"geometry": ["POINT (1 1)"]})
        result = df.with_columns(wkt_to_shape_col(pl.col("geometry")).alias("shape"))
        self.assertIsInstance(result["shape"][0], Point)

    def test_geojson_to_wkt_col(self):
        df = pl.DataFrame({"geometry": [{'type': 'Point', 'coordinates': [1, 1]}]})
        result = df.with_columns(geojson_to_wkt_col(pl.col("geometry")).alias("wkt"))
        self.assertEqual(result["wkt"][0], "POINT (1 1)")

    def test_shape_to_geoalchemy2_col(self):
        df = pl.DataFrame({"geometry": [Point(1, 1)]})
        result = df.with_columns(shape_to_geoalchemy2_col(pl.col("geometry")).alias("geoalchemy2"))
        self.assertTrue(result["geoalchemy2"][0].startswith("0101000000"))

    def test_geoalchemy2_to_shape_col(self):
        df = pl.DataFrame({"geometry": ["0101000000000000000000F03F0000000000000040"]})  # WKB for POINT (1 2)
        result = df.with_columns(geoalchemy2_to_shape_col(pl.col("geometry")).alias("shape"))
        self.assertIsInstance(result["shape"][0], Point)
        self.assertEqual(result["shape"][0], Point(1, 2))

    def test_wkt_to_geoalchemy_col(self):
        df = pl.DataFrame({"geometry": ["POINT (1 1)"]})
        result = df.with_columns(wkt_to_geoalchemy_col(pl.col("geometry"), 4326, 2056).alias("geoalchemy2"))
        self.assertTrue(result["geoalchemy2"][0].startswith("0101000000"))

    def test_geoalchemy2_to_wkt_col(self):
        df = pl.DataFrame({"geometry": ["0101000000000000000000F03F0000000000000040"]})  # WKB for POINT (1 2)
        result = df.with_columns(geoalchemy2_to_wkt_col(pl.col("geometry"), 2056, 4326).alias("wkt"))
        self.assertTrue(result["wkt"][0].startswith("POINT (-19.917"))

if __name__ == "__main__":
    unittest.main()
