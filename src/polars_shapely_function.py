from typing import Optional, Union
from shapely import LineString, from_wkt,buffer, intersects, union_all, Geometry
from shapely.ops import transform
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, shape, MultiLineString

from itertools import batched


import polars as pl
from polars import col as c
from pyproj import CRS, Transformer

from shapely_function import (
    shape_to_geoalchemy2, geoalchemy2_to_shape, point_list_to_linestring, shape_coordinate_transformer
)

def shape_intersect_shape_col(geo_str: pl.Expr, geometry: Geometry) -> pl.Expr:
    """
    Check if geometries in a Polars expression intersect with a given polygon.

    Args:
        geo_str (pl.Expr): The Polars expression containing geometries in WKT format.
        polygon (Polygon): The polygon to check for intersection.

    Returns:
        pl.Expr: A Polars expression with boolean values indicating intersection.
    """
    return geo_str.pipe(wkt_to_shape_col).map_elements(lambda x: intersects(x, geometry), return_dtype=pl.Boolean)


def shape_intersect_polygon(geo_str: pl.Expr, polygon: Polygon) -> pl.Expr:
    """
    Check if geometries in a Polars expression intersect with a given polygon.

    Args:
        geo_str (pl.Expr): The Polars expression containing geometries in WKT format.
        polygon (Polygon): The polygon to check for intersection.

    Returns:
        pl.Expr: A Polars expression with boolean values indicating intersection.
    """
    return geo_str.pipe(wkt_to_shape_col).map_elements(lambda x: intersects(x, polygon), return_dtype=pl.Boolean)

def get_linestring_boundaries_col(line_str: pl.Expr) -> pl.Expr:
    """
    Get the boundary nodes of geometries in a Polars expression.

    Args:
        line_str (pl.Expr): The Polars expression containing linestring in WKT format.

    Returns:
        pl.Expr: A Polars expression with lists of boundary nodes in WKT format.
    """
    return (
        line_str.pipe(wkt_to_shape_col).map_elements(
        lambda x: list(map(lambda point: point.wkt, x.boundary.geoms)), return_dtype=pl.List(pl.Utf8))
        )

def get_geometry_list(df: pl.DataFrame, col_name: str = "geometry") -> list[Union[Point, LineString, Polygon]]:
    """
    Get a list of geometries from a Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        col_name (str, optional): The name of the geometry column. Defaults to "geometry".

    Returns:
        list[Union[Point, LineString, Polygon]]: The list of geometries.
    """
    return df.select(c(col_name).pipe(wkt_to_shape_col))[col_name].to_list()

def get_multigeometry_from_col(
    df: pl.DataFrame, col_name: str = "geometry"
    ) -> Union[MultiPoint, MultiLineString, MultiPolygon]:
    """
    Get a MultiGeometry object from a Polars DataFrame column.

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        col_name (str, optional): The name of the geometry column. Defaults to "geometry".

    Returns:
        Union[MultiPoint, MultiLineString, MultiPolygon]: The MultiGeometry object.
    """
    geo_list: list[Union[Point, LineString, Polygon]] = get_geometry_list(df=df, col_name=col_name)
    if isinstance(geo_list[0], Point):
        return MultiPoint(geo_list) # type: ignore
    elif isinstance(geo_list[0], LineString):
        return MultiLineString(geo_list) # type: ignore
    else:
        return MultiPolygon(geo_list) # type: ignore
    
def add_buffer(geo_str: pl.Expr, buffer_size: float) -> pl.Expr:
    """
    Add a buffer to geometries in a Polars expression.

    Args:
        geo_str (pl.Expr): The Polars expression containing geometries in WKT format.
        buffer_size (float): The buffer size.

    Returns:
        pl.Expr: A Polars expression with buffered geometries in WKT format.
    """
    return geo_str.pipe(wkt_to_shape_col).map_elements(lambda x: buffer(x, buffer_size).wkt, return_dtype=pl.Utf8)


def calculate_line_length(line_str: pl.Expr) -> pl.Expr:
    """
    Calculate the length of LineString geometries in a Polars expression.

    Args:
        line_str (pl.Expr): The Polars expression containing LineString geometries in WKT format.

    Returns:
        pl.Expr: A Polars expression with the lengths of the LineString geometries.
    """
    return line_str.pipe(wkt_to_shape_col).map_elements(lambda x: x.length, return_dtype=pl.Float64)

def shape_coordinate_transformer_col(shape_col: pl.Expr, srid_from: int, srid_to: int) -> pl.Expr:
    """
    Transform the coordinates of geometries in a Polars expression from one CRS to another.

    Args:
        shape_col (pl.Expr): The Polars expression containing geometries.
        srid_from (int): The source spatial reference system identifier.
        srid_to (int): The target spatial reference system identifier.

    Returns:
        pl.Expr: A Polars expression with transformed geometries.
    """
    return shape_col.map_elements(
        lambda x: shape_coordinate_transformer(x, srid_from=srid_from, srid_to=srid_to), return_dtype=pl.Object)

def generate_point_from_coordinates(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Generate Point geometries from x and y coordinates in Polars expressions.

    Args:
        x (pl.Expr): The Polars expression containing x coordinates.
        y (pl.Expr): The Polars expression containing y coordinates.

    Returns:
        pl.Expr: A Polars expression with Point geometries in WKT format.
    """
    return (
        pl.concat_list([x, y]).map_elements(lambda coord: Point(*coord).wkt, return_dtype=pl.Utf8)
    )

def generate_shape_linestring(coord_list: pl.Expr) -> pl.Expr:
    """
    Generate LineString geometries from coordinate lists in a Polars expression.

    Args:
        coord_list (pl.Expr): The Polars expression containing coordinate lists.

    Returns:
        pl.Expr: A Polars expression with LineString geometries.
    """
    return (
        coord_list.map_elements(lambda x: LineString(batched(x, 2)).wkt, return_dtype=pl.Utf8)
    )

def get_linestring_from_point_list(point_list_str: pl.Expr) ->  pl.Expr:
    """
    Generate LineString geometries from point lists in a Polars expression.

    Args:
        point_list_str (pl.Expr): The Polars expression containing point lists in WKT format.

    Returns:
        pl.Expr: A Polars expression with LineString geometries in WKT format.
    """
    return point_list_str.map_elements(point_list_to_linestring, return_dtype=pl.Utf8)

def combine_shape(geometry_list_str: pl.Expr) ->  pl.Expr:
    """
    Combine multiple geometries into a single geometry in a Polars expression.

    Args:
        geometry_list_str (pl.Expr): The Polars expression containing geometry lists in WKT format.

    Returns:
        pl.Expr: A Polars expression with combined geometries in WKT format.
    """
    return geometry_list_str.map_elements(lambda x: union_all(list(map(from_wkt, x))).wkt, return_dtype=pl.Utf8)


def shape_to_wkt_col(geometry: pl.Expr) ->  pl.Expr:
    """
    Convert geometries in a Polars expression to WKT format.

    Args:
        geometry (pl.Expr): The Polars expression containing geometries.

    Returns:
        pl.Expr: A Polars expression with geometries in WKT format.
    """
    return geometry.map_elements(lambda x: x.wkt, return_dtype=pl.Utf8)

def wkt_to_shape_col(geometry: pl.Expr) ->  pl.Expr:
    """
    Convert WKT strings in a Polars expression to geometries.

    Args:
        geometry (pl.Expr): The Polars expression containing WKT strings.

    Returns:
        pl.Expr: A Polars expression with geometries.
    """
    return geometry.map_elements(from_wkt, return_dtype=pl.Object)

def geojson_to_wkt_col(geometry: pl.Expr) ->  pl.Expr:
    """
    Convert GeoJSON strings in a Polars expression to WKT format.

    Args:
        geometry (pl.Expr): The Polars expression containing GeoJSON strings.

    Returns:
        pl.Expr: A Polars expression with geometries in WKT format.
    """
    return geometry.map_elements(lambda x: shape(x).wkt, return_dtype=pl.Utf8)

def shape_to_geoalchemy2_col(geo: pl.Expr) -> pl.Expr:
    """
    Convert geometries in a Polars expression to GeoAlchemy2 format.

    Args:
        geo (pl.Expr): The Polars expression containing geometries.

    Returns:
        pl.Expr: A Polars expression with geometries in GeoAlchemy2 format.
    """
    return geo.map_elements(shape_to_geoalchemy2, return_dtype=pl.Utf8)

def geoalchemy2_to_shape_col(geo_str: pl.Expr) -> pl.Expr:
    """
    Convert GeoAlchemy2 strings in a Polars expression to geometries.

    Args:
        geo_str (pl.Expr): The Polars expression containing GeoAlchemy2 strings.

    Returns:
        pl.Expr: A Polars expression with geometries.
    """
    return geo_str.map_elements(geoalchemy2_to_shape, return_dtype=pl.Object)

def wkt_to_geoalchemy_col(geo_str: pl.Expr, srid_from: Optional[int], srid_to: Optional[int]) -> pl.Expr:
    """
    Convert WKT strings in a Polars expression to GeoAlchemy2 format.

    Args:
        geo_str (pl.Expr): The Polars expression containing WKT strings.
        srid_from (Optional[int]): The source spatial reference system identifier.
        srid_to (Optional[int]): The target spatial reference system identifier.

    Returns:
        pl.Expr: A Polars expression with geometries in GeoAlchemy2 format.

    Raises:
        ValueError: If only one of srid_from or srid_to is provided.
    """
    if (srid_from is None) and (srid_to is None):
        return (
            geo_str.pipe(wkt_to_shape_col)
            .pipe(shape_to_geoalchemy2_col)
        )
    elif (srid_from is not None) and (srid_to is not None):
        return (
            geo_str
            .pipe(wkt_to_shape_col)
            .pipe(shape_coordinate_transformer_col, srid_from=srid_from, srid_to=srid_to)
            .pipe(shape_to_geoalchemy2_col)
        )
    else:
        raise ValueError("Both srid_from and srid_to must be provided or None.")

def geoalchemy2_to_wkt_col(geo_str: pl.Expr, srid_from: Optional[int], srid_to: Optional[int]) -> pl.Expr:
    """
    Convert GeoAlchemy2 strings in a Polars expression to WKT format.

    Args:
        geo_str (pl.Expr): The Polars expression containing GeoAlchemy2 strings.
        srid_from (Optional[int]): The source spatial reference system identifier.
        srid_to (Optional[int]): The target spatial reference system identifier.

    Returns:
        pl.Expr: A Polars expression with geometries in WKT format.

    Raises:
        ValueError: If only one of srid_from or srid_to is provided.
    """
    if (srid_from is None) and (srid_to is None):
        return (
            geo_str
            .pipe(geoalchemy2_to_shape_col)
            .pipe(shape_to_wkt_col)
        )
    elif (srid_from is not None) and (srid_to is not None):
        return (
            geo_str
            .pipe(geoalchemy2_to_shape_col)
            .pipe(shape_coordinate_transformer_col, srid_from=srid_from, srid_to=srid_to)
            .pipe(shape_to_wkt_col)
        )
    else:
        raise ValueError("Both srid_from and srid_to must be provided or None.")