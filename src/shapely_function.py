import os
import json

from typing import Optional, Union
from shapely import Geometry, LineString, from_wkt, intersection, distance, buffer, intersects, convex_hull
from shapely.ops import nearest_points, split, snap, linemerge, transform
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, shape
from shapely.prepared import prep
import numpy as np 

from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.elements import WKBElement
from math import ceil

SWISS_SRID = 2056
GPS_SRID = 4326

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
from shapely.ops import transform
from pyproj import CRS, Transformer

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
