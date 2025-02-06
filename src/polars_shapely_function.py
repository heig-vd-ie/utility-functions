from typing import Optional, Union
from shapely import LineString, from_wkt,buffer, intersects, union_all, Geometry, extract_unique_points
from shapely.ops import transform, nearest_points
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, shape, MultiLineString

from itertools import batched


import polars as pl
from polars import col as c
from pyproj import CRS, Transformer

from shapely_function import (
    shape_to_geoalchemy2, geoalchemy2_to_shape, point_list_to_linestring, shape_coordinate_transformer,
    get_multipoint_from_wkt_list, get_multilinestring_from_wkt_list, get_nearest_point_within_distance
)


def get_coordinates_list_from_col(df: pl.DataFrame, col_name: str = "geometry") -> list[tuple[float, float]]:
    """
    Extract a list of coordinates from a specified column containing geometric data.

    Args:
        df (pl.DataFrame): The DataFrame to process.
        col_name (str): The name of the column containing geometric data.

    Returns:
        list: A list of coordinates extracted from the specified column.
    """
    point_list: list[Point] = list(extract_unique_points(get_multigeometry_from_col(df=df, col_name=col_name)).geoms)
    return list(map(lambda point: (point.x, point.y), point_list))

def get_coordinates_col(col: pl.Expr) -> pl.Expr:
    """
    Add a new column to the DataFrame with coordinates extracted from a specified column containing geometric data.

    Args:
        col (pl.Expr): The column containing geometric data.

    Returns:
        polars.DataFrame: The DataFrame with an additional column of coordinates.
    """
    return col.pipe(wkt_to_shape_col).map_elements(lambda x: list(x.coords), return_dtype=pl.List(pl.List(pl.Float64)))

def get_nearest_point_within_distance_col(point: pl.Expr, point_list: MultiPoint, min_distance: float) -> pl.Expr:
    """
    Find the nearest point within a specified minimum distance from each point in a column containing geometric data.

    Args:
        point (pl.Expr): The expression representing the column containing the reference points.
        point_list (MultiPoint): The MultiPoint geometry containing the points to search.
        min_distance (float): The minimum distance to search for the nearest point.

    Returns:
        pl.Expr: An expression with the nearest points within the specified minimum distance.

    Examples:
    ~~~~~~~~~~
    
    >>> data = {'geometry': [Point(0, 0).wkt, Point(2, 2).wkt, Point(4, 4).wkt]}
    ... df = pl.DataFrame(data)
    ... multi_point = MultiPoint([Point(1, 1), Point(10, 1), Point(2, 1)])
    ... df.with_column(get_nearest_point_within_distance_col(pl.col('geometry'), multi_point, 2).alias('nearest_point'))
    shape: (3, 2)
    ┌─────────────────┬──────────────────┐
    │ geometry        ┆ nearest_point    │
    │ ---             ┆ ---              │
    │ str             ┆ str              │
    ╞═════════════════╪══════════════════╡
    │ POINT (0 0)     ┆ POINT (1 1)      │
    │ POINT (2 2)     ┆ POINT (2 1)      │
    │ POINT (4 4)     ┆ None             │
    └─────────────────┴──────────────────┘
    """
    return (
        point
        .pipe(wkt_to_shape_col) 
        .map_elements(
            lambda x: get_nearest_point_within_distance(point=x, point_list=point_list, min_distance=min_distance), 
            return_dtype=pl.Utf8)
        )
    

def get_multipoint_from_wkt_list_col(point_list: pl.Expr) -> pl.Expr:
    """
    Convert a column containing WKT representations of lines into a Multipoint geometry.

    Args:
        point_list (pl.Expr): The expression representing the column containing list of Point.

    Returns:
        pl.Expr: An expression with MultiPoint geometries.
    """
    return point_list.map_elements(get_multipoint_from_wkt_list, pl.Object)

def get_multilinestring_from_wkt_list_col(linestring_list: pl.Expr) -> pl.Expr:
    """
    Convert a column containing WKT representations of lines into a MultiLineString geometry.

    Args:
        linestring_list (pl.Expr): The expression representing the column containing list of linestring.

    Returns:
        pl.Expr: An expression with MultiLineString geometries.
    """
    return linestring_list.map_elements(get_multilinestring_from_wkt_list, pl.Object)

def get_point_list_centroid_col(point_list: pl.Expr) -> pl.Expr:
    """
    Calculate the centroid of a list of points from a specified column containing geometric data.

    Args:
        point_list (pl.Expr): The expression representing the column containing the list of points.

    Returns:
        pl.Expr: An expression with the centroid points.

    Example:
    
    >>> data = {'points': [Point(0, 0).wkt, Point(1, 1).wkt, Point(2, 2).wkt]}
    ... df = pl.DataFrame(data)
    ... df.with_column(get_point_list_centroid_col(pl.col('points')).alias('centroid'))
    shape: (1, 1)
    ┌────────────────┐
    │ centroid       │
    │ ---            │
    │ str            │
    ╞════════════════╡
    │ POINT (1 1)    │
    └────────────────┘
    """    
    return (
        point_list.pipe(get_multipoint_from_wkt_list_col)
        .map_elements(lambda x: x.centroid.wkt, return_dtype=pl.Utf8)
    )
    
def generate_linestring_from_nearest_points_col(point: pl.Expr, multi_point: MultiPoint):
    """
    Generate a LineString geometry from the nearest points within a specified distance in a column containing geometric data.

    Args:
        point (pl.Expr): The expression representing the column containing the reference points.
        multi_point (MultiPoint): The MultiPoint geometry containing the points to search.

    Returns:
        pl.Expr: An expression with LineString geometries created from the nearest points.

    Example:
    
    >>> data = {'geometry': [Point(0, 0).wkt, Point(2, 2).wkt]}
    ... df = pl.DataFrame(data)
    ... multi_point = MultiPoint([Point(0, 1), Point(1, 1), Point(2, 1)])
    ... df.with_column(
    ...     c(geometry).pipe(generate_linestring_from_nearest_points_col, multi_point=multi_point)
    ...     .alias('linestring')
    ... )
    shape: (3, 2)
    ┌──────────────────┬─────────────────────────┐
    │ geometry         ┆ linestring              │
    │ ---              ┆ ---                     │
    │ str              ┆ str                     │
    ╞══════════════════╪═════════════════════════╡
    │ POINT (0 0)      ┆ LINESTRING (0 0, 0 1)   │
    │ POINT (2 2)      ┆ LINESTRING (2 2, 2 1)   │
    └──────────────────┴─────────────────────────┘
    """
    return (
        point.pipe(wkt_to_shape_col)
        .map_elements(lambda x: LineString(nearest_points(x, multi_point)).wkt, return_dtype=pl.Utf8)
    )
    
def linestring_is_ring_col(linestring: pl.Expr) -> pl.Expr:
    """
    Check if the LineString geometries in a specified column form a closed ring.

    Args:
        linestring (pl.Expr): The expression representing the column containing LineString geometries.

    Returns:
        pl.Expr: An expression indicating whether each LineString is a closed ring.

    Example:
    
    >>> data = {
    ...     'geometry': [
    ...         LineString([(0, 0), (1, 1), (1, 0), (0, 0)]).wkt, 
    ...         LineString([(0, 0), (1, 1), (1, 0)]).wkt
    ...     ]}
    ... df = pl.DataFrame(data)
    ... df.with_column(linestring_is_ring_col(pl.col('geometry')).alias('is_ring'))
    shape: (2, 2)
    ┌───────────────────────────────────────┬──────────┐
    │ geometry                              ┆ is_ring  │
    │ ---                                   ┆ ---      │
    │ str                                   ┆ bool     │
    ╞═══════════════════════════════════════╪══════════╡
    │ LINESTRING (0 0, 1 1, 1 0, 0 0)       ┆ true     │
    │ LINESTRING (0 0, 1 1, 1 0)            ┆ false    │
    └───────────────────────────────────────┴──────────┘
    """
    
    return linestring.pipe(wkt_to_shape_col).map_elements(lambda x: x.is_closed, return_dtype=pl.Boolean)


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

def generate_linestring_from_coordinates_list(coord_list: pl.Expr) -> pl.Expr:
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