"""
Auxiliary functions
"""
# import dotenv
from typing import Optional
import logging
import os
import uuid
import coloredlogs
import polars as pl
from polars import col as c

import re 
import owncloud
import tqdm
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkt

from config import settings

NAMESPACE_UUID: uuid.UUID = uuid.UUID('{bc4d4e0c-98c9-11ec-b909-0242ac120002}')

def scan_switch_directory(
    oc: owncloud.Client, local_folder_path: str, switch_folder_path: str, download_anyway: bool) -> list[str]:
    file_list = []
    build_non_existing_dirs(os.path.join(local_folder_path, switch_folder_path))
    for file_data in oc.list(switch_folder_path): # type: ignore
        file_path: str = file_data.path
        if file_data.file_type == "dir":
            file_list.extend(scan_switch_directory(
                oc=oc, local_folder_path=local_folder_path, 
                switch_folder_path=file_path[1:], download_anyway=download_anyway))
        else:
            if (not os.path.exists(local_folder_path + file_path)) | download_anyway:
                file_list.append(file_path)
    return file_list

def download_from_switch(switch_folder_path: str, local_folder_path: str= ".cache", download_anyway: bool = False):
    oc: owncloud.Client = owncloud.Client.from_public_link(settings.SWITCH_LINK, folder_password=settings.SWITCH_PASS)
    with tqdm.tqdm(total = 1, desc=f"Scan {switch_folder_path} Switch remote directory", ncols=120) as pbar:
        file_list: list[str] = scan_switch_directory(
            oc=oc, local_folder_path=local_folder_path, 
            switch_folder_path=switch_folder_path, download_anyway=download_anyway)
        pbar.update()
    for file_path in tqdm.tqdm(
        file_list, desc= f"Download files from {switch_folder_path} Switch remote directory ", ncols=120
        ):
        oc.get_file(file_path, local_folder_path + file_path)
        

def generate_log(name: str) -> logging.Logger:
    """
    load configs of environment
    :return: log for logging, config is list of all env vars
    """
    log = logging.getLogger(name)
    coloredlogs.install(level="info")
    return log


def build_non_existing_dirs(file_path: str):
    """
    build non existing directories
    :param file_path:
    :return: True
    """
    file_path = os.path.normpath(file_path)
    # Split the path into individual directories
    dirs = file_path.split(os.sep)
    # Check if each directory exists and create it if it doesn't
    current_path = ""
    for directory in dirs:
        current_path = os.path.join(current_path, directory)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
    return True


def pl_to_dict(df: pl.DataFrame) -> dict:
    """
    Convert a Polars DataFrame with two columns into a dictionary.

    Args:
        df (pl.DataFrame): Polars DataFrame with two columns.

    Returns:
        dict: Dictionary representation of the DataFrame.
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame is not composed of two columns")

    columns_name = df.columns[0]
    df = df.drop_nulls(columns_name)
    if df[columns_name].is_duplicated().sum() != 0:
        raise ValueError("Key values are not unique")
    return dict(df.to_numpy())

def modify_string(string: str, format_str: dict) -> str:
    """
    Modify a string by replacing substrings according to a format dictionary.

    Args:
        string (str): Input string.
        format_str (dict): Dictionary containing the substrings to be replaced and their replacements.

    Returns:
        str: Modified string.
    """

    for str_in, str_out in format_str.items():
        string = re.sub(str_in, str_out, string)
    return string

def camel_to_snake(s):
    return (
        ''.join(
            [ '_'+ c.lower() if c.isupper() else c for c in s ]
        ).lstrip('_')
    )

def snake_to_camel(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def convert_list_to_string(list_data)-> str:
    return ", ".join(map(str, list_data))

def table_to_gpkg(table: pl.DataFrame, gpkg_file_name: str, layer_name: str, ):
    list_columns: list[str] = [name for name, col_type in dict(table.schema).items() if type(col_type) == pl.List]
    table_pd: pd.DataFrame = table.with_columns(
        c(list_columns).list.join(", ")
    ).to_pandas()

    table_pd["geometry"] = table_pd["geometry"].apply(from_wkt)
    table_pd = table_pd[table_pd.geometry.notnull()]
    table_gpd: gpd.GeoDataFrame = gpd.GeoDataFrame(
        table_pd.dropna(axis=0, subset="geometry"), crs=settings.SWISS_SRID) # type: ignore
    table_gpd = table_gpd[~table_gpd["geometry"].is_empty] # type: ignore
    table_gpd.to_file(gpkg_file_name, layer=layer_name) 


def dict_to_gpkg(data: dict, file_path:str):

    with tqdm.tqdm(range(1), ncols=100, desc="Save input data in gpkg format") as pbar:
        for layer_name, table in data.items():
            if isinstance(table, pl.DataFrame):
                if not table.is_empty():
                    table_to_gpkg(table=table, gpkg_file_name=file_path, layer_name=layer_name)
        pbar.update()



# def filter_unique_nodes_from_list(node_id_list: pl.Expr)-> pl.Expr:
    
#     return (
#         pl.when(node_id_list.list.len() == 1)
#         .then(node_id_list.list.get(0, null_on_oob=True))
#         .otherwise(pl.lit(None))
#     )


def dictionary_key_filtering(dictionary: dict, key_list: list) -> dict:
    return dict(filter(lambda x : x[0] in key_list, dictionary.items()))


def generate_uuid(base_value: str, base_uuid: uuid.UUID | None = None, added_string: str = "") -> str:
    """
    Generate a UUID based on a base value, base UUID, and an optional added string.

    Args:
        base_value (str): The base value for generating the UUID.
        base_uuid (str): The base UUID for generating the UUID.
        added_string (str, optional): The optional added string. Defaults to "".

    Returns:
        str: The generated UUID.
    """
    if base_uuid is None:
        base_uuid=NAMESPACE_UUID
    return str(uuid.uuid5(base_uuid, added_string + base_value))



