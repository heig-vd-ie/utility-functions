from ast import List
from operator import ge
import os
import json
from itertools import chain
from copy import deepcopy

import re
from typing import Optional, Union
from shapely import (
    Geometry, LineString, from_wkt, intersection, distance, buffer, intersects, convex_hull,
    extract_unique_points, line_merge, intersection_all)
from shapely import transform as sh_transform

from shapely.ops import nearest_points, split, snap, linemerge, transform, substring
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, shape, MultiLineString
from shapely.prepared import prep
import numpy as np 


from pyproj import CRS, Transformer

from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.elements import WKBElement
from math import ceil

SWISS_SRID = 2056
GPS_SRID = 4326


def get_point_side(line: LineString, point: Point) -> Optional[str]:
    """
    Determine which side of a line a point is on.

    Args:
        point (Point): The point to check.
        line (LineString): The line to check against.

    Returns:
        str: `start` if the point is to the beginning of the line, 1 if to the right, 
        and `end` if it is at the end.
    """
    boundaries = list(line.boundary.geoms)
    if point == boundaries[0]:
        return "start"
    elif point == boundaries[1]:
        return "end"
    else:
        return None

def get_nearest_point_within_distance(point: Point, point_list: MultiPoint, min_distance: float) -> Optional[str]:
    """
    Find the nearest point within a specified distance from a given point.

    Args:
        point (Point): The reference point.
        point_list (MultiPoint): The list of points to search.
        min_distance (float): The minimum distance to search for the nearest point.

    Returns:
        str or None: The nearest point in wkt format within the specified distance, or None if no 
        point is found.
    """    
    nearest_points_list = nearest_points(point_list, point)
    if distance(*nearest_points_list) < min_distance:
        return nearest_points_list[0].wkt
    return None

def get_point_list_centroid(point_list: list[Point]) -> Point:
    """
    Calculate the centroid of a list of points.

    Args:
        points (Point): The list of points.

    Returns:
        Point: The centroid of the list of points.
    """
    return MultiPoint(point_list).centroid

def get_multipoint_from_wkt_list(point_list: list[str]) -> MultiPoint:
    """
    Convert a list of WKT representations of points into a MultiPoint geometry.

    Args:
        point_list (list[str])): The list of WKT points strings.

    Returns:
        MultiPoint: The MultiPoint geometry.
    """
    return MultiPoint(list(map(from_wkt, point_list))) # type: ignore

def get_multilinestring_from_wkt_list(linestring_list: list[str]) -> MultiPoint:
    """
    Convert a list of WKT representations of linestring into a MultiLineString geometry.

    Args:
        linestring_list (list[str]): The list of WKT linestring strings.
    Returns:
        MultiLineString: The MultiLineString geometry.
    """
    return MultiLineString(list(map(from_wkt, linestring_list))) # type: ignore

def get_multipolygon_from_wkt_list(polygon_list: list[str]) -> MultiPolygon:
    """
    Convert a list of WKT representations of polygons into a MultiPolygon geometry.

    Args:
        polygon_list (list[str])): The list of WKT polygons strings.

    Returns:
        MultiPolygon: The MultiPolygon geometry.
    """
    return MultiPolygon(list(map(from_wkt, polygon_list))) # type: ignore


def point_list_to_linestring(point_list_str: list[str]) -> str:
    """
    Convert a list of WKT point strings to a WKT LineString.

    Args:
        point_list_str (list[str]): The list of WKT point strings.

    Returns:
        str: The WKT LineString.
    """
    return LineString(list(map(from_wkt, point_list_str))).wkt # type: ignore


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

def shape_to_geoalchemy2(geo: Geometry, srid: int = GPS_SRID) -> str:
    """
    Convert a Shapely Geometry object to a GeoAlchemy2 WKBElement string.

    Args:
        geo (Geometry): The Shapely Geometry object.
        srid (int, optional): The spatial reference system identifier. Defaults to GPS_SRID = 4326.

    Returns:
        str: The GeoAlchemy2 WKBElement string.
    """
    if isinstance(geo, Geometry):
        return from_shape(geo, srid=srid).desc
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
    polygon: Polygon = list(polygon_str.wkt.geoms)[0] # type: ignore
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
    nx = int(ceil((maxx - minx)/delta)) + 1
    ny = int(ceil((maxy - miny)/delta)) + 1
    gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
    grid = []
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
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

def generate_valid_polygon(multipolygon_str: str) -> Optional[Union[Polygon, MultiPolygon]]:
    """
    Generate a valid polygon from a MultiPolygon WKT string.

    Args:
        multipolygon_str (str): The WKT string of the MultiPolygon.

    Returns:
        Optional[Polygon]: The valid polygon.
    """
    shape: Geometry = from_wkt(multipolygon_str)
    if isinstance(shape, MultiPolygon):
        return MultiPolygon(list(
            map(
                lambda x: x if x.is_valid 
                else x.convex_hull, 
                shape.geoms
            )) # type: ignore
        )
    elif isinstance(shape, Polygon):
        return shape if shape.is_valid else convex_hull(shape) # type: ignore
    else:
        return None


def shape_coordinate_transformer(shape: Geometry, srid_from: int, srid_to: int) -> Geometry:
    """
    Transform the coordinates of geometries from one CRS to another.

    Args:
        shape (Geometry): The Polars expression containing geometries.
        srid_from (int): The source spatial reference system identifier.
        srid_to (int): The target spatial reference system identifier.

    Returns:
        pl.Expr: A Polars expression with transformed geometries.
    """
    transformer = Transformer.from_crs(
        crs_from=CRS(f"EPSG:{srid_from}"), crs_to=CRS(f"EPSG:{srid_to}"), always_xy=True).transform
    return transform(transformer, shape)
    

def load_shape_from_geo_json(
        file_name: str, srid_from: Optional[str] = None, srid_to: Optional[str]= None
    ) -> Geometry:
    """
    Load a shape from a GeoJSON file and optionally transform its coordinates.

    Args:
        file_name (str): The path to the GeoJSON file.
        srid_from (Optional[str], optional): The source spatial reference system identifier. Defaults to None.
        srid_to (Optional[str], optional): The target spatial reference system identifier. Defaults to None.

    Returns:
        Geometry: The loaded shape.

    Raises:
        FileNotFoundError: If the GeoJSON file is not found.
        ValueError: If only one of srid_from or srid_to is provided.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    with open(file_name) as f:
        loading_shape = json.load(f)
    geo_shape: Geometry = shape(loading_shape["features"][0]["geometry"])

    if (srid_from is None) | (srid_to is None):
        return geo_shape
    elif (srid_from is not None) and (srid_to is not None):
        return shape_coordinate_transformer(geo_shape, srid_from=srid_from, srid_to=srid_to)  # type: ignore
    else:
        raise ValueError("Both srid_from and srid_to must be provided or None.")

def segment_list_from_multilinestring(multi_linestring: MultiLineString) -> list[LineString]:
    """
    Generate a list of LineString segments from a MultiLineString.

    Args:
        multi_linestring (MultiLineString): The MultiLineString object.

    Returns:
        list[LineString]: The list of LineString segments.
    """
    segments: MultiLineString = intersection(multi_linestring, multi_linestring) # type: ignore
    return list(segments.geoms)  # type: ignore

def shape_list_to_wkt_list(shape_list: list[Geometry]) -> list[str]:
    """
    Convert a list of Shapely Geometry objects to a list of WKT strings.

    Args:
        shape_list (list[Geometry]): The list of Shapely Geometry objects.

    Returns:    
        list[str]: The list of WKT strings.
    """
    return list(map(lambda x: x.wkt, shape_list)) # type: ignore

def wkt_list_to_shape_list(str_list: list[str]) -> list[Geometry]:
    """
    Convert a list of Shapely Geometry objects to a list of WKT strings.

    Args:
        shape_list (list[Geometry]): The list of Shapely Geometry objects.

    Returns:    
        list[str]: The list of WKT strings.
    """
    return list(map(from_wkt, str_list)) # type: ignore

def multipoint_from_multilinestring(multilinestring: MultiLineString) -> MultiPoint:
    """
    Generate a MultiPoint from a MultiLineString by extracting unique points.

    Args:
        multilinestring (MultiLineString): The MultiLineString object.

    Returns:
        MultiPoint: The MultiPoint object containing unique points from the MultiLineString.
    """
    return MultiPoint(extract_unique_points(multilinestring))

def move_geometry(data: dict) -> str:
    return (
        sh_transform(
            geometry = from_wkt(data["geometry"]),
            transformation=lambda x: x + np.array([np.cos(-data["angle"]), np.sin(-data["angle"])])*data["distance"], # type: ignore
        ).wkt
    )
    
def merge_multilinestring_creating_missing_segments(
    mls: Optional[Union[MultiLineString, LineString]]) -> Optional[LineString]:
    """
    Merge a MultiLineString into a single LineString, creating missing segments if necessary.

    Args:
        mls (Union[MultiLineString, LineString]): The MultiLineString or LineString to merge.

    Returns:
        LineString: The merged LineString.

    Raises:
        ValueError: If the merging process fails.
    """
    if isinstance(mls, LineString):
        return mls
    if mls is None:
        return None
    multipoint_list= []
    for line in mls.geoms:
            multipoint_list.append(list(line.boundary.geoms))
    new_segment = []    
    for i, multipoint in enumerate(multipoint_list): 
        remaining_list = deepcopy(multipoint_list)
        remaining_list.pop(i)
        new_nearest_points = nearest_points(g1=MultiPoint(multipoint), g2=MultiPoint(list(chain(*remaining_list))))
        new_segment.append(LineString(sorted(LineString(new_nearest_points).coords)))
        
    new_linestring = line_merge(MultiLineString(list(mls.geoms) + list(set(new_segment))))
    # print(new_linestring)
    if isinstance(new_linestring, MultiLineString):
        new_linestring= merge_multilinestring_creating_missing_segments(new_linestring)
    if isinstance(new_linestring, LineString):
        return new_linestring
    
    raise ValueError("Error in merge_multilinestring_creating_missing_segments")


def linestring_splitter(linestring_str: Optional[str], nb_split: int) -> Optional[list[str]]:
    """
    Split LineString in WKT format into a list of LineString with the same length.

    Args:
        linestring_str (Optional]): The LineString to split in in WKT format.
        nb_split (int): The number of segments to split the LineString into.
    Returns:
        List[str]: The list of the splitted LineString.
    """
    if linestring_str is None:
        return None
    line: Geometry = from_wkt(linestring_str)
    if not isinstance(line, LineString):
        return None
    sub_line: list[str] = list(map( 
            lambda x : 
            substring(line, start_dist=x*line.length/nb_split, end_dist=(x+1)*line.length/nb_split).wkt, range(nb_split)
    ))
    return sub_line


def get_geometry_list_intersection(geometry_list: list) -> Optional[str]:
    """
    Get the intersection of a list of geometries in WKT format.
    Args:
        geometry_list (list): The list of geometries in WKT format.
    Returns:
        str: The WKT string of the intersection of the geometries.
    """
    
    multi_line= get_multilinestring_from_wkt_list(geometry_list)
    multi_line = intersection_all(multi_line.geoms) # type: ignore
    if isinstance(multi_line, MultiLineString):
        multi_line = linemerge(multi_line)
    if multi_line.is_empty:
        return None
    return multi_line.wkt



def remove_linestring_circle_from_multilinestring(shape_str: str) -> str:
    """
    Remove the circle in the multilinestring and merge results
    Args:
        shape_str (str): WKT representation of the geometry
    Returns:
        str: WKT representation of the geometry without the circle
    1. Convert the WKT string to a Shapely geometry object.
    2. Check if the geometry is a MultiLineString.
    3. If it is, filter out the circular linestrings (those that are rings).
    4. Merge the remaining linestrings using the `line_merge` function.
    5. Return the WKT representation of the merged geometry.
    """
    multi_line: Geometry = from_wkt(shape_str)
    if isinstance(multi_line, MultiLineString):
        return line_merge(MultiLineString(list(filter(lambda x: not x.is_ring, multi_line.geoms)))).wkt # type: ignore
    else:
        return shape_str