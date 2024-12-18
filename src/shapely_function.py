
"""
This module provides various utility functions for working with geometric shapes using the Shapely library.
Functions:
    get_polygon_list(geo_shape: Geometry) -> list[str]:
        Returns a list of WKT strings representing the polygons in the given geometry.
    point_list_to_linestring(point_list_str: list[str]) -> str:
        Converts a list of WKT point strings to a WKT LineString.
    get_polygon_multipoint_intersection(polygon_str: str, multipoint: MultiPoint) -> Optional[list[str]]:
        Returns a list of WKT strings representing the intersection points between a polygon and a multipoint.
    find_closest_node(data: dict, node_id_geo_mapping: dict) -> Optional[str]:
        Finds the closest node ID from a given node ID geometry mapping.
    explode_multipolygon(geometry_str: str) -> list[Polygon]:
        Explodes a MultiPolygon WKT string into a list of Polygon objects.
    get_geo_multipoints(data_df: pl.DataFrame, column_name: str = "geometry") -> MultiPoint:
        Converts a DataFrame column of WKT strings to a MultiPoint object.
    get_closest_point(geo_str: str, multi_point: MultiPoint, max_distance: float=100) -> Optional[str]:
        Finds the closest point within a maximum distance from a given geometry WKT string.
    linestring_from_node_str(node_str: list[str]) -> str:
        Converts a list of WKT node strings to a WKT LineString.
    calculate_line_length(line_str: pl.Expr) -> pl.Expr:
        Calculates the length of a LineString expression.
    filter_geo_from_polygon(geo: pl.Expr, polygon: Polygon) -> pl.Expr:
        Filters geometries that intersect with a given polygon.
    get_branch_node(geo: pl.Expr) -> pl.Expr:
        Returns the boundary points of a geometry as a list of WKT strings.
    get_geometry_list(df: pl.DataFrame) -> list[Geometry]:
        Converts a DataFrame column of WKT strings to a list of Geometry objects.
    add_buffer(geo: pl.Expr, buffer_size: float) -> pl.Expr:
        Adds a buffer to a geometry expression and returns the buffered geometry as a WKT string.
    geoalchemy2_to_shape(geo_str: str) -> Geometry:
        Converts a GeoAlchemy2 WKBElement string to a Shapely Geometry object.
    shape_to_geoalchemy2(geo: Geometry) -> str:
        Converts a Shapely Geometry object to a GeoAlchemy2 WKBElement string.
"""
from typing import Optional
from shapely import Geometry, LineString, from_wkt, intersection, distance, buffer, intersects
from shapely.ops import nearest_points
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString

from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.elements import WKBElement

from config import settings


def point_list_to_linestring(point_list_str: list[str]) -> str:
    return LineString(list(map(from_wkt, point_list_str))).wkt


def get_polygon_multipoint_intersection(polygon_str: str, multipoint: MultiPoint) -> Optional[list[str]]:
    point_shape: Geometry = intersection(from_wkt(polygon_str), multipoint)
    
    if isinstance(point_shape, MultiPoint):
        return list(map(lambda x: x.wkt, point_shape.geoms))
    if isinstance(point_shape, Point):
        if point_shape.is_empty:
            return []
        return [point_shape.wkt]
    return []

def find_closest_node_from_list(data: dict, node_name: str, node_list_name: str) -> Optional[str]:
    if data["node_id"] is None:
        return None  
    if len(data[node_list_name]) == 0:
        return None  
    if len(data[node_list_name]) == 1:
        return data[node_list_name][0] 
    if len(data[node_list_name]) > 1:
        return min(data[node_list_name], key=lambda x: from_wkt(data[node_name]).distance(from_wkt(x)))
    return None

def explode_multipolygon(geometry_str: str) -> list[Polygon]:
    geometry_shape: Geometry = from_wkt(geometry_str)
    if isinstance(geometry_shape, Polygon):
        return [geometry_shape]
    if isinstance(geometry_shape, MultiPolygon):
        return list(geometry_shape.geoms)
    return []

    
def geoalchemy2_to_shape(geo_str: str) -> Geometry:
    return to_shape(WKBElement(str(geo_str)))

def shape_to_geoalchemy2(geo: Geometry) -> str:
    if isinstance(geo, Geometry):
        return from_shape(geo, srid=settings.GPS_SRID).desc
    return None


    
