from typing import Optional
from shapely import Geometry, LineString, from_wkt, intersection, distance, buffer, intersects, union_all
from shapely.ops import nearest_points, transform
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString

from itertools import batched

from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.elements import WKBElement

import polars as pl
from polars import col as c
from pyproj import CRS, Transformer

from shapely_function import shape_to_geoalchemy2, geoalchemy2_to_shape, point_list_to_linestring

from config import settings


def shape_intersect_polygon(geo: pl.Expr, polygon: Polygon) -> pl.Expr:
    return geo.map_elements(lambda x: intersects(from_wkt(x), polygon), return_dtype=pl.Boolean)

def get_branch_node_col(geo: pl.Expr) -> pl.Expr:
    return geo.map_elements(
        lambda x: list(map(lambda point: point.wkt, from_wkt(x).boundary.geoms)), return_dtype=pl.List(pl.Utf8)
    )

def get_geometry_list(df: pl.DataFrame) -> list[Geometry]:
    return (
        df.select(
            c("geometry").map_elements(from_wkt, return_dtype=pl.Object)
        )["geometry"].to_list()
    )
    
def add_buffer(geo: pl.Expr, buffer_size: float) -> pl.Expr:
    return geo.map_elements(lambda x: buffer(from_wkt(x), buffer_size).wkt, return_dtype=pl.Utf8)

def get_closest_point_from_multi_point(geo_str: str, multi_point: MultiPoint, max_distance: float=100) -> Optional[str]:
    geo = from_wkt(geo_str)
    _, closest_point = nearest_points(geo, multi_point)
    if distance(geo, closest_point) < max_distance:
        return closest_point.wkt
    return None

def linestring_str_from_node_str(node_str: list[str]) -> str:
    return LineString(list(map(from_wkt, node_str))).wkt


def calculate_line_length(line_str: pl.Expr) -> pl.Expr:
    return line_str.map_elements(lambda x: from_wkt(x).length, return_dtype=pl.Float64)



def shape_to_geoalchemy2_col(geo: pl.Expr) -> pl.Expr:
    return geo.map_elements(shape_to_geoalchemy2, return_dtype=pl.Utf8)

def geoalchemy2_to_shape_col(geo_str: pl.Expr) -> pl.Expr:
    return geo_str.map_elements(geoalchemy2_to_shape, return_dtype=pl.Object)

def shape_coordinate_transformer_col(shape_col: pl.Expr, crs_from: str, crs_to: str) -> pl.Expr:
    transformer = Transformer.from_crs(crs_from=CRS(crs_from), crs_to=CRS(crs_to) , always_xy=True).transform
    return shape_col.map_elements(lambda x: transform(transformer, x), return_dtype=pl.Object)

def generate_shape_point(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return (
        pl.concat_list([x, y]).map_elements(lambda coord: Point(*coord), return_dtype=pl.Object)
    )

def wkt_to_geoalchemy(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.map_elements(from_wkt, return_dtype=pl.Object)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.SWISS_SRID, crs_to=settings.GPS_SRID)
        .pipe(shape_to_geoalchemy2_col)
    )

def geoalchemy2_to_wkt(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.pipe(geoalchemy2_to_shape_col)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.GPS_SRID, crs_to=settings.SWISS_SRID)
        .map_elements(lambda x: x.wkt, return_dtype=pl.Utf8)
    )


def generate_geo_linestring(coord_list: pl.Expr) -> pl.Expr:

    return (
        coord_list.map_elements(lambda x: LineString(batched(x, 2)), return_dtype=pl.Object)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.SWISS_SRID, crs_to=settings.GPS_SRID)
        .pipe(shape_to_geoalchemy2_col)
    )

def combine_shape(geo_shape: pl.Expr) ->  pl.Expr:
    return geo_shape.map_elements(lambda x: union_all(list(map(from_wkt, x))).wkt, return_dtype=pl.Utf8)

def point_list_to_linestring_col(point_list: pl.Expr) -> pl.Expr:
    return point_list.map_elements(point_list_to_linestring, return_dtype=pl.Utf8)
