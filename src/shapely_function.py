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
from shapely import Geometry, LineString, from_wkt, intersection, distance, buffer, intersects, convex_hull
from shapely.ops import nearest_points, split, snap, linemerge, transform
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString
from shapely.prepared import prep
import numpy as np 

from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.elements import WKBElement

from config import settings


def point_list_to_linestring(point_list_str: list[str]) -> str:
    """
    Convert a list of WKT point strings to a WKT LineString.

    Args:
        point_list_str (list[str]): The list of WKT point strings.

    Returns:
        str: The WKT LineString.
    """
    return LineString(list(map(from_wkt, point_list_str))).wkt


def get_polygon_multipoint_intersection(polygon_str: str, multipoint: MultiPoint) -> Optional[list[str]]:
    """
    Get the intersection points between a polygon and a multipoint.

    Args:
        polygon_str (str): The WKT string of the polygon.
        multipoint (MultiPoint): The MultiPoint object.

    Returns:
        Optional[list[str]]: The list of WKT strings representing the intersection points.
    """
    point_shape: Geometry = intersection(from_wkt(polygon_str), multipoint)
    
    if isinstance(point_shape, MultiPoint):
        return list(map(lambda x: x.wkt, point_shape.geoms))
    if isinstance(point_shape, Point):
        if point_shape.is_empty:
            return []
        return [point_shape.wkt]
    return []

def find_closest_node_from_list(data: dict, node_name: str, node_list_name: str) -> Optional[str]:
    """
    Find the closest node ID from a given node ID geometry mapping.

    Args:
        data (dict): The data dictionary containing node information.
        node_name (str): The key for the node ID.
        node_list_name (str): The key for the list of node IDs.

    Returns:
        Optional[str]: The closest node ID.
    """
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
    """
    Explode a MultiPolygon WKT string into a list of Polygon objects.

    Args:
        geometry_str (str): The WKT string of the MultiPolygon.

    Returns:
        list[Polygon]: The list of Polygon objects.
    """
    geometry_shape: Geometry = from_wkt(geometry_str)
    if isinstance(geometry_shape, Polygon):
        return [geometry_shape]
    if isinstance(geometry_shape, MultiPolygon):
        return list(geometry_shape.geoms)
    return []

    
def geoalchemy2_to_shape(geo_str: str) -> Geometry:
    """
    Convert a GeoAlchemy2 WKBElement string to a Shapely Geometry object.

    Args:
        geo_str (str): The GeoAlchemy2 WKBElement string.

    Returns:
        Geometry: The Shapely Geometry object.
    """
    return to_shape(WKBElement(str(geo_str)))

def shape_to_geoalchemy2(geo: Geometry) -> str:
    """
    Convert a Shapely Geometry object to a GeoAlchemy2 WKBElement string.

    Args:
        geo (Geometry): The Shapely Geometry object.

    Returns:
        str: The GeoAlchemy2 WKBElement string.
    """
    if isinstance(geo, Geometry):
        return from_shape(geo, srid=settings.GPS_SRID).desc
    return None

def get_closest_point_from_multi_point(geo_str: str, multi_point: MultiPoint, max_distance: float=100) -> Optional[str]:
    """
    Find the closest point within a maximum distance from a given geometry WKT string.

    Args:
        geo_str (str): The WKT string of the geometry.
        multi_point (MultiPoint): The MultiPoint object.
        max_distance (float, optional): The maximum distance. Defaults to 100.

    Returns:
        Optional[str]: The WKT string of the closest point.
    """
    geo = from_wkt(geo_str)
    _, closest_point = nearest_points(geo, multi_point)
    if distance(geo, closest_point) < max_distance:
        return closest_point.wkt
    return None
    
def remove_z_coordinates(geom: Geometry)->Geometry:
    """
    Remove the Z coordinates from a geometry.

    Args:
        geom (Geometry): The Shapely Geometry object.

    Returns:
        Geometry: The Shapely Geometry object without Z coordinates.
    """
    return transform(lambda x, y, z=None: (x, y), geom)

def get_valid_polygon_str(polygon_str: dict) -> str:
    """
    Get a valid polygon WKT string from a dictionary.

    Args:
        polygon_str (dict): The dictionary containing polygon information.

    Returns:
        str: The valid polygon WKT string.
    """
    polygon: Polygon = list(shape(polygon_str).geoms)[0] # type: ignore
    if polygon.is_valid:
        return polygon.wkt
    return polygon.convex_hull.wkt

def grid_bounds(geom, delta):
    """
    Generate a grid of polygons within the bounds of a geometry.

    Args:
        geom (Geometry): The Shapely Geometry object.
        delta (float): The grid cell size.

    Returns:
        list[Polygon]: The list of grid polygons.
    """
    minx, miny, maxx, maxy = geom.bounds
    nx = int((maxx - minx)/delta)
    ny = int((maxy - miny)/delta)
    gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
    grid = []
    for i in range(len(gx)-1):
        for j in range(len(gy)-1):
            poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])
            grid.append( poly_ij )
    return grid

def partition(geom: Polygon, delta: float) -> list:
    """
    Partition a polygon into smaller polygons based on a grid.

    Args:
        geom (Polygon): The Shapely Polygon object.
        delta (float): The grid cell size.

    Returns:
        list[Polygon]: The list of partitioned polygons.
    """
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, delta)))
    return grid

def generate_valid_polygon(multipolygon_str: str) -> Optional[Polygon]:
    """
    Generate a valid polygon from a MultiPolygon WKT string.

    Args:
        multipolygon_str (str): The WKT string of the MultiPolygon.

    Returns:
        Optional[Polygon]: The valid polygon.
    """
    shape: Geometry = from_wkt(multipolygon_str)
    if isinstance(shape, MultiPolygon):
        shape = shape.geoms[0]
    elif isinstance(shape, Polygon):
        pass
    else:
        return None
    if shape.is_valid:
        return shape
    return convex_hull(shape) # type: ignore
