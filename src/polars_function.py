import re
import uuid
import json
from datetime import timedelta, datetime

from typing import Optional

import polars as pl
from polars import col as c
import numpy as np

from general_function import modify_string, generate_log, generate_uuid

from config import settings

# Global variable
log = generate_log(name=__name__)

def generate_uuid_col(col: pl.Expr, base_uuid: Optional[uuid.UUID] = None, added_string: str = "") -> pl.Expr:
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
        "1": True, "true": True , "oui": True, "0": False, 
        "false": False, "vrai": True, "non": False, 
        "off": False, "on": True}
    return col.str.to_lowercase().replace(format_str, default=False).cast(pl.Boolean)

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
    )


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

