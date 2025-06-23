import re
import uuid
import json
from datetime import timedelta, datetime

from typing import Optional, Union

import polars as pl
from polars import col as c
import numpy as np

from general_function import modify_string, generate_log, generate_uuid


# Global variable
log = generate_log(name=__name__)

def cum_count_duplicates(cols_names: Union[str, list[str]]) -> pl.Expr:
    """
    Calculate the cumulative count of duplicate values in a specified column of a DataFrame, 
    assigning half of the count as strict positive values and the other half as strict negative values.

    Parameters:
        cols_names (Union[str, list[str]]): The name of the column to check for duplicates.

    Returns:
        pl.Expr: A polar expression showing the cumulative count of duplicates.
        
    Example:
    ~~~~~~~~
    
    >>> df = pl.DataFrame({"a": [1, 1, 2, 3, 4, 4, 4]})
    ... df.with_columns(
    ...    cum_count_duplicates(cols_names="a").alias("cum_count")
    ... )
    shape: (7, 2)
    ┌─────┬────────────┐
    │ id  ┆ cum_count  │
    │ --- ┆ ---        │
    │ i64 ┆ i64        │
    ╞═════╪════════════╡
    │ 1   ┆ -1         │
    │ 1   ┆ 1          │
    │ 2   ┆ 1          │
    │ 3   ┆ 1          │
    │ 4   ┆ -1         │
    │ 4   ┆ 1          │
    │ 4   ┆ 2          │
    └─────┴────────────┘
    
    """
    if isinstance(cols_names, str):
        cols_names = [cols_names]
    cum_count_col: pl.Expr  = (
        c(cols_names[0]).cum_count().cast(pl.Int32) - c(cols_names[0]).count() // 2 -1).over(cols_names)
    
    return pl.when(cum_count_col < 0).then(cum_count_col).otherwise(cum_count_col+1)

def generate_uuid_col(
    col: pl.Expr, base_uuid: Optional[uuid.UUID] = None, added_string: str = "") -> pl.Expr:
    """
    Generate UUIDs for a column based on a base UUID and an optional added string.

    Args:
        col (pl.Expr): The column to generate UUIDs for.
        base_uuid (uuid.UUID, optional): The base UUID for generating the UUIDs.
        added_string (str, optional): The optional added string. Defaults to "".

    Returns:
        pl.Expr: The column with generated UUIDs.
    """

    return (
        col.cast(pl.Utf8)
        .map_elements(lambda x: generate_uuid(base_value=x, base_uuid=base_uuid, added_string=added_string), pl.Utf8)
    )

def cast_float(float_str: pl.Expr) -> pl.Expr:
    """
    Cast a string column to float, modifying the string format as needed.

    Args:
        float_str (pl.Expr): The string column to cast.

    Returns:
        pl.Expr: The casted float column.
    """
    format_str = {r'^,': "0.", ',': "."}
    return float_str.pipe(modify_string_col, format_str=format_str).cast(pl.Float64)

def cast_boolean(col: pl.Expr) -> pl.Expr:
    """
    Cast a column to boolean based on predefined replacements.

    Args:
        col (pl.Expr): The column to cast.

    Returns:
        pl.Expr: The casted boolean column.
    """
    format_str = {
        "1": True, "true": True , "oui": True, "1.0": True, "0": False, "0.0": False, 
        "false": False, "vrai": True, "non": False, 
        "off": False, "on": True}
    return col.cast(pl.Utf8).str.to_lowercase().replace_strict(format_str, default=False).cast(pl.Boolean)

def modify_string_col(string_col: pl.Expr, format_str: dict) -> pl.Expr:
    """
    Modify string columns based on a given format dictionary.

    Args:
        string_col (pl.Expr): The string column to modify.
        format_str (dict): The format dictionary containing the string modifications.

    Returns:
        pl.Expr: The modified string column.
    """
    return (
        string_col.map_elements(
            lambda x: modify_string(string=x, format_str=format_str), return_dtype=pl.Utf8, skip_nulls=True)
    )

def parse_date(date_str: Optional[str], default_date: datetime) -> datetime:
    """
    Parse a date string and return a datetime object.

    Args:
        date_str (str, optional): The date string to parse.
        default_date (datetime): The default date to return if the date string is None.

    Returns:
        datetime: The parsed datetime object.
    
    Raises:
        ValueError: If the date format is not recognized.
    """
    if date_str is None:
        return default_date
    if bool(re.match(r"[0-9]{5}", date_str)):
        return  datetime(1899, 12, 30) + timedelta(days=int(date_str))

    format_str: dict[str, str] = {r"[-:.//]": "_"}
    date_str = modify_string(date_str, format_str)
    if bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}", date_str)):
        return datetime.strptime(date_str, '%Y_%m_%d')
    if bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}", date_str)):
        return datetime.strptime(date_str, '%d_%m_%Y')
    
    raise ValueError("Date format not recognized")


def parse_timestamp(
        timestamp_str: pl.Expr, item: Optional[str],  
        keep_string_format: bool= False, convert_to_utc: bool = False, 
        initial_time_zone: str = "Europe/Zurich"
    ) -> pl.Expr:
    """
    Parse a timestamp column based on a given item.

    Args:
        timestamp_str (pl.Expr): The timestamp column.
        item (str, optional): The item to parse.
        keep_string_format (bool, optional): Whether to keep the string format. Defaults to False.
        convert_to_utc (bool, optional): Whether to convert the timestamp to UTC. Defaults to False.
        initial_time_zone (str, optional): The initial time zone of the timestamps. Defaults to "Europe/Zurich".
    Returns:
        pl.Expr: The parsed timestamp column.

    Raises:
        ValueError: If the timestamp format is not recognized.
    """
    format_str: dict[str, str] = {r"[-:\.//]": "_"}

    if item is None:
        return pl.lit(None)
    item = modify_string(item, format_str)
    if bool(re.match(r"[0-9]{5}", item)):
        timestamp: pl.Expr =  (3.6e6*24*timestamp_str.cast(pl.Int32)).cast(pl.Duration("ms")) +  datetime(1899, 12, 30)
    else:
        if bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}\s[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{3}", item)):
            format_str: dict[str, str] = { r"[-:.//]": "_", r"_[0-9]{3}$": ""}
            format_timestamp: str = "%d_%m_%Y %H_%M_%S"
        elif bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}\s[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{3}", item)):
            format_str = { r"[-:.//]": "_", r"_[0-9]{3}$": ""}
            format_timestamp: str = "%Y_%m_%d %H_%M_%S"
        elif bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}\s[0-9]{2}_[0-9]{2}_[0-9]{2}", item)):
            format_timestamp: str = "%Y_%m_%d %H_%M_%S"
        elif bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}\s[0-9]{2}_[0-9]{2}_[0-9]{2}", item)):
            format_timestamp: str ="%d_%m_%Y %H_%M_%S"
        elif bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{2}\s[0-9]{2}_[0-9]{2}_[0-9]{2}", item)):
            format_timestamp: str ="%d_%m_%y %H_%M_%S"
        elif bool(re.match(r"[0-9]{4}_[0-9]{2}_[0-9]{2}", item)):
            timestamp_str = timestamp_str + " 00_00_00"
            format_timestamp: str ="%Y_%m_%d %H_%M_%S"
        elif bool(re.match(r"[0-9]{2}_[0-9]{2}_[0-9]{4}", item)):
            timestamp_str = timestamp_str + " 00_00_00"
            format_timestamp: str ="%d_%m_%Y %H_%M_%S"
        else:
            raise ValueError("Timestamp format not recognized")
    
        timestamp: pl.Expr = (
            modify_string_col(timestamp_str, format_str)
            .str.strptime(pl.Datetime, format_timestamp)
            .dt.cast_time_unit(time_unit="us")  
        )
    if keep_string_format:
        return timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    elif convert_to_utc:
        return timestamp.pipe(cast_to_utc_timestamp, initial_time_zone=initial_time_zone)
    return timestamp

def cast_to_utc_timestamp(timestamp: pl.Expr, initial_time_zone: str = "Europe/Zurich") -> pl.Expr:
    """
    Convert a timestamp column to UTC from the specified initial time zone.

    Args:
        timestamp (pl.Expr): The timestamp column to convert.
        initial_time_zone (str, optional): The initial time zone of the timestamps. Defaults to "Europe/Zurich".

    Returns:
        pl.Expr: The timestamp column converted to UTC.
    """
    return (
        pl.when(timestamp.is_first_distinct())
        .then(timestamp.dt.replace_time_zone(initial_time_zone, ambiguous='earliest'))
        .otherwise(timestamp.dt.replace_time_zone(initial_time_zone, ambiguous='latest'))
        .dt.convert_time_zone("UTC")
    )

def generate_random_uuid(col: pl.Expr) -> pl.Expr:
    """
    Generate a random UUID.

    Returns:
        str: The generated UUID.
    """
    return col.map_elements(lambda x: str(uuid.uuid4()), return_dtype=pl.Utf8, skip_nulls=False)


def get_meta_data_string(metadata: pl.Expr) -> pl.Expr:
    """
    Convert metadata to a JSON string, excluding keys with None values.

    Args:
        metadata (pl.Expr): The metadata column.

    Returns:
        pl.Expr: The metadata column as JSON strings.
    """
    return (
        metadata.map_elements(
            lambda x: json.dumps({key: value for key, value in x.items() if value is not None}, ensure_ascii=False), 
        return_dtype=pl.Utf8)
    ).replace({"{}": None})


def digitize_col(col: pl.Expr, min: float, max: float, nb_state: int) -> pl.Expr:
    """
    Digitize a column into discrete states based on the specified number of states.

    Args:
        col (pl.Expr): The column to digitize.
        min (float): The minimum value of the column.
        max (float): The maximum value of the column.
        nb_state (int): The number of discrete states.

    Returns:
        pl.Expr: The digitized column.
    """
    bins = np.linspace(min, max, nb_state + 1)
    return (
        col.map_elements(lambda x: np.digitize(x, bins), return_dtype=pl.Int64)
    )


def get_transfo_impedance(rated_v: pl.Expr, rated_s: pl.Expr, voltage_ratio: pl.Expr) -> pl.Expr:
    """
    Get the transformer impedance (or resistance if real part) based on the short-circuit tests.

    Args:
        rated_v (pl.Expr): The rated voltage column indicates which side of the transformer the parameters are 
        associated with (usually lv side).[V].
        rated_s (pl.Expr): The rated power column [VA].
        voltage_ratio (pl.Expr): The ratio between the applied input voltage to get rated current when transformer 
        secondary is short-circuited and the rated voltage [%].

    Returns:
        pl.Expr: The transformer impedance column [Ohm].
    """
    return voltage_ratio  / 100 * (rated_v**2)/ rated_s

def get_transfo_admittance(rated_v: pl.Expr, rated_s: pl.Expr, oc_current_ratio: pl.Expr) -> pl.Expr:
    """
    Get the transformer admittance based on the open circuit test
    
    Args:
        rated_v (pl.Expr): The rated voltage column indicates which side of the transformer the parameters are 
        associated with (usually lv side).[V].
        rated_s (pl.Expr): The rated power column [VA].
        oc_current_ratio (pl.Expr): The ratio between the measured current when transformer secondary is opened and the
        rated current [%].

    Returns:
        pl.Expr: The transformer admittance column [Simens].
    """
    return oc_current_ratio / 100 * rated_s / (rated_v **2)

def get_transfo_conductance(rated_v: pl.Expr, iron_losses: pl.Expr) -> pl.Expr:
    """
    Get the transformer conductance based on iron losses measurement.

    Args:
        rated_v (pl.Expr): The rated voltage column indicates which side of the transformer the parameters are 
        associated with (usually lv side).[V].
        iron_losses (pl.Expr): The iron losses column [W].

    Returns:
        pl.Expr: The transformer conductance column [Simens].
    """
    return  iron_losses /(rated_v**2)

def get_transfo_resistance(rated_v: pl.Expr, rated_s: pl.Expr, copper_losses: pl.Expr) -> pl.Expr:
    """
    Get the transformer resistance based on copper losses measurement.

    Args:
        rated_v (pl.Expr): The rated voltage column indicates which side of the transformer the parameters are 
        associated with (usually lv side).[V].
        rated_s (pl.Expr): The rated power column [VA].
        copper_losses (pl.Expr): The copper losses column [W].

    Returns:
        pl.Expr: The transformer resistance column [Ohm].
    """
    return  copper_losses * ((rated_v/rated_s)**2)

def get_transfo_imaginary_component(module: pl.Expr, real: pl.Expr) -> pl.Expr:
    """
    Get the transformer imaginary component based on the module and real component.

    Args:
        module (pl.Expr): The module column [Ohm or Simens].
        real (pl.Expr): The real component column [Ohm or Simens].

    Returns:
        pl.Expr: The transformer imaginary component column [Ohm or Simens].
    """
    return (np.sqrt(module ** 2 - real ** 2))

def concat_list_of_list(col_list: pl.Expr) -> pl.Expr:
    """
    Concatenate a column of lists into a list containing sublist.

    Args:
        col_list (pl.Expr): The column of lists to concatenate.

    Returns:
        pl.Expr: The concatenated list column.
    """
    return pl.concat_list(
        col_list.map_elements(lambda x: [x], return_dtype=pl.List(pl.List(pl.Float64)))
    )


def linear_interpolation_for_bound(x_col: pl.Expr, y_col: pl.Expr) -> pl.Expr:
    """
    Perform linear interpolation for boundary values in a column.

    Args:
        x_col (pl.Expr): The x-axis column.
        y_col (pl.Expr): The y-axis column to interpolate.

    Returns:
        pl.Expr: The interpolated y-axis column.
    """
    a_diff: pl.Expr = y_col.diff()/x_col.diff()
    x_diff: pl.Expr = x_col.diff().backward_fill()
    y_diff: pl.Expr = pl.coalesce(
        pl.when(y_col.is_null().or_(y_col.is_nan()))
        .then(a_diff.forward_fill()*x_diff)
        .otherwise(pl.lit(0)).cum_sum(),
        pl.when(y_col.is_null().or_(y_col.is_nan()))
        .then(-a_diff.backward_fill()*x_diff)
        .otherwise(pl.lit(0)).cum_sum(reverse=True)
    )

    return y_col.backward_fill().forward_fill() + y_diff

def linear_interpolation_using_cols(
    df: pl.DataFrame, x_col: str, y_col: Union[list[str], str]
    ) -> pl.DataFrame:
    """
    Perform linear interpolation on specified columns of a DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the x-axis column.
        y_col (Union[list[str], str]): The name(s) of the y-axis column(s) to interpolate.

    Returns:
        pl.DataFrame: The DataFrame with interpolated y-axis columns.
    """
    df = df.sort(x_col)
    x = df[x_col].to_numpy()
    if isinstance(y_col, str):
        y_col = [y_col]
    for col in y_col:
        y = df[col].to_numpy()
        mask = ~np.isnan(y)
        df = df.with_columns(
            pl.Series(np.interp(x, x[mask], y[mask], left=np.nan, right=np.nan)).fill_nan(None).alias(col)
        ).with_columns(
            linear_interpolation_for_bound(x_col=c(x_col), y_col=c(col)).alias(col)
        )
    return df

def replace_null_list(
    col: pl.Expr, default_value: Optional[Union[list, str, int, float]] = None
    ) -> pl.Expr:
    """
    Replace null values in a list column with a specified value.

    Args:
        col (pl.Expr): The list column to modify.
        default_value (Optional[Union[list, str, int, float]], optional): The default value for nulls. Defaults to None.

    Returns:
        pl.Expr: The modified list column.
    """
    return pl.when(col == []).then(default_value).otherwise(col)

def list_to_list_of_tuple(list_col: pl.Expr) -> pl.Expr:
    """
    Convert a list of lists to a list of tuples.
    Args:
        list_col: A polars expression representing a list of lists.
    Returns:
        A polars expression representing a list of tuples.
    """
    return list_col.map_elements(lambda x: [tuple(x)], return_dtype=pl.List(pl.Object))

def keep_only_duplicated_list(data: pl.Expr) -> pl.Expr:
    """
    Return a boolean Polars expression indicating which rows in a list column are duplicates,
    after sorting and joining the list elements with an underscore.
    This function is useful for identifying rows in a DataFrame where the concatenated list of elements
    contains duplicates no mater the position of the elements.
    Args:
        data (pl.Expr): A Polars expression representing a list column.
    Returns:
        pl.Expr: A boolean Polars expression indicating whether the concatenated list of elements is duplicated
    """
    return data.cast(pl.List(pl.Utf8)).list.sort().list.join(separator="_").is_duplicated()
