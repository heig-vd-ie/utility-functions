from typing import Optional, Union
from shapely import Geometry, LineString, from_wkt, intersection, distance, buffer, intersects, union_all
from shapely.ops import nearest_points, transform
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, shape, MultiLineString

from itertools import batched

from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.elements import WKBElement

import polars as pl
from polars import col as c
from pyproj import CRS, Transformer

from shapely_function import shape_to_geoalchemy2, geoalchemy2_to_shape, point_list_to_linestring

from config import settings


def shape_intersect_polygon(geo_str: pl.Expr, polygon: Polygon) -> pl.Expr:
    return geo_str.pipe(wkt_to_shape_col).map_elements(lambda x: intersects(x, polygon), return_dtype=pl.Boolean)

def get_branch_node_col(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.pipe(wkt_to_shape_col).map_elements(
        lambda x: list(map(lambda point: point.wkt, x.boundary.geoms)), return_dtype=pl.List(pl.Utf8))
        )

def get_geometry_list(df: pl.DataFrame, col_name: str = "geometry") -> list[Union[Point, LineString, Polygon]]:
    
    return df.select(c(col_name).pipe(wkt_to_shape_col))[col_name].to_list()

def get_multigeometry_from_col(df: pl.DataFrame, col_name: str = "geometry") -> Union[MultiPoint, MultiLineString, MultiPolygon]:
    geo_list: list[Union[Point, LineString, Polygon]] = get_geometry_list(df=df, col_name=col_name)
    if isinstance(geo_list[0], Point):
        return MultiPoint(geo_list) # type: ignore
    elif isinstance(geo_list[0], LineString):
        return MultiLineString(geo_list) # type: ignore
    else:
        return MultiPolygon(geo_list) # type: ignore
    
def add_buffer(geo_str: pl.Expr, buffer_size: float) -> pl.Expr:
    return geo_str.pipe(wkt_to_shape_col).map_elements(lambda x: buffer(x, buffer_size).wkt, return_dtype=pl.Utf8)


def calculate_line_length(line_str: pl.Expr) -> pl.Expr:
    return line_str.pipe(wkt_to_shape_col).map_elements(lambda x: x.length, return_dtype=pl.Float64)

def shape_coordinate_transformer_col(shape_col: pl.Expr, crs_from: str, crs_to: str) -> pl.Expr:
    transformer = Transformer.from_crs(crs_from=CRS(crs_from), crs_to=CRS(crs_to) , always_xy=True).transform
    return shape_col.map_elements(lambda x: transform(transformer, x), return_dtype=pl.Object)

def generate_point_from_coordinates(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return (
        pl.concat_list([x, y]).map_elements(lambda coord: Point(*coord), return_dtype=pl.Object)
    )

def generate_shape_linestring(coord_list: pl.Expr) -> pl.Expr:

    return (
        coord_list.map_elements(lambda x: LineString(batched(x, 2)), return_dtype=pl.Object)
    )

def get_linestring_from_point_list(point_list_str: pl.Expr) ->  pl.Expr:
    return point_list_str.map_elements(point_list_to_linestring, return_dtype=pl.Utf8)

def combine_shape(geometry_list_str: pl.Expr) ->  pl.Expr:
    return geometry_list_str.map_elements(lambda x: union_all(list(map(from_wkt, x))).wkt, return_dtype=pl.Utf8)


def shape_to_wkt_col(geometry: pl.Expr) ->  pl.Expr:
    return geometry.map_elements(lambda x: x.wkt, return_dtype=pl.Utf8)

def wkt_to_shape_col(geometry: pl.Expr) ->  pl.Expr:
    return geometry.map_elements(from_wkt, return_dtype=pl.Object)

def geojson_to_wkt_col(geometry: pl.Expr) ->  pl.Expr:
    return geometry.map_elements(lambda x: shape(x).wkt, return_dtype=pl.Utf8)

def shape_to_geoalchemy2_col(geo: pl.Expr) -> pl.Expr:
    return geo.map_elements(shape_to_geoalchemy2, return_dtype=pl.Utf8)

def geoalchemy2_to_shape_col(geo_str: pl.Expr) -> pl.Expr:
    return geo_str.map_elements(geoalchemy2_to_shape, return_dtype=pl.Object)

def wkt_to_geoalchemy_col(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.map_elements(from_wkt, return_dtype=pl.Object)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.SWISS_SRID, crs_to=settings.GPS_SRID)
        .pipe(shape_to_geoalchemy2_col)
    )

def geoalchemy2_to_wkt_col(geo_str: pl.Expr) -> pl.Expr:
    return (
        geo_str.pipe(geoalchemy2_to_shape_col)
        .pipe(shape_coordinate_transformer_col, crs_from=settings.GPS_SRID, crs_to=settings.SWISS_SRID)
        .map_elements(lambda x: x.wkt, return_dtype=pl.Utf8)
    )