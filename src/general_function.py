"""
Auxiliary functions
"""
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

NAMESPACE_UUID: uuid.UUID = uuid.UUID('{bc4d4e0c-98c9-11ec-b909-0242ac120002}')

def scan_switch_directory(
    oc: owncloud.Client, local_folder_path: str, switch_folder_path: str, download_anyway: bool) -> list[str]:
    """
    Scan a directory on the SWITCH server and return a list of file paths.

    Args:
        oc (owncloud.Client): The ownCloud client.
        local_folder_path (str): The local folder path.
        switch_folder_path (str): The SWITCH folder path.
        download_anyway (bool): Whether to download files even if they already exist locally.

    Returns:
        list[str]: List of file paths.
    """
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

def download_from_switch(
    switch_folder_path: str, switch_link: str, switch_pass: str, local_folder_path: str= ".cache", 
    download_anyway: bool = False):
    """
    Download files from a SWITCH directory to a local folder.

    Args:
        switch_folder_path (str): The SWITCH folder path.
        switch_link (str): The public link to the SWITCH folder.
        switch_pass (str): The password for the SWITCH folder.
        local_folder_path (str, optional): The local folder path. Defaults to ".cache".
        download_anyway (bool, optional): Whether to download files even if they already exist locally. Defaults to False.
    """
    oc: owncloud.Client = owncloud.Client.from_public_link(public_link=switch_link, folder_password=switch_pass)
    with tqdm.tqdm(total = 1, desc=f"Scan {switch_folder_path} Switch remote directory", ncols=120) as pbar:
        file_list: list[str] = scan_switch_directory(
            oc=oc, local_folder_path=local_folder_path, 
            switch_folder_path=switch_folder_path, download_anyway=download_anyway)
        pbar.update()
    for file_path in tqdm.tqdm(
        file_list, desc= f"Download files from {switch_folder_path} Switch remote directory ", ncols=120
        ):
        oc.get_file(file_path, local_folder_path + file_path)
        

def generate_log(name: str, log_level: str= "info") -> logging.Logger:
    """
    Generate a logger with the specified name and log level.

    Args:
        name (str): The name of the logger.
        log_level (str, optional): The log level. Defaults to "info".

    Returns:
        logging.Logger: The generated logger.
    """
    log = logging.getLogger(name)
    coloredlogs.install(level=log_level)
    return log


def build_non_existing_dirs(file_path: str):
    """
    Build non-existing directories for a given file path.

    Args:
        file_path (str): The file path.

    Returns:
        bool: True if directories were created successfully.
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
    Convert a Polars DataFrame with two columns into a dictionary. It is assumed that the 
    first column contains the keys and the second column contains the values. The keys must 
    be unique but Null values will be filtered.

    Args:
        df (pl.DataFrame): Polars DataFrame with two columns.

    Returns:
        dict: Dictionary representation of the DataFrame.

    Raises:
        ValueError: If the DataFrame does not have exactly two columns or if the keys are not unique.
    """
    
    if df.shape[1] != 2:
        raise ValueError("DataFrame is not composed of two columns")

    columns_name = df.columns[0]
    df = df.drop_nulls(columns_name)
    if df[columns_name].is_duplicated().sum() != 0:
        raise ValueError("Key values are not unique")
    return dict(df.rows())

def modify_string(string: str, format_str: dict) -> str:
    """
    Modify a string by replacing substrings according to a format dictionary 
    -   Input could contains RegEx.
    -   The replacement is done in the order of the dictionary keys.

    Args:
        string (str): Input string.
        format_str (dict): Dictionary containing the substrings to be replaced and their replacements.

    Returns:
        str: Modified string.
    """

    for str_in, str_out in format_str.items():
        string = re.sub(str_in, str_out, string)
    return string

def camel_to_snake(camel_str: str) -> str:
    """
    Convert a camelCase string to snake_case.

    Args:
        camel_str (str): The camelCase string.

    Returns:
        str: The snake_case string.
    """
    return (
        ''.join(
            [ '_'+ c.lower() if c.isupper() else c for c in camel_str ]
        ).lstrip('_')
    )

def snake_to_camel(snake_str: str) -> str:
    """
    Convert a snake_case string to CamelCase.

    Args:
        snake_str (str): The snake_case string.

    Returns:
        str: The CamelCase string.
    """
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def convert_list_to_string(list_data: list) -> str:
    """
    Convert a list to a comma-separated string.

    Args:
        list_data (list): The list to convert.

    Returns:
        str: The comma-separated string.
    """
    return ", ".join(map(str, list_data))

def table_to_gpkg(table: pl.DataFrame, gpkg_file_name: str, layer_name: str):
    """
    Save a Polars DataFrame as a GeoPackage file. As GeoPackage does not support list columns, 
    the list columns are joined into a single string separated with a comma.

    Args:
        table (pl.DataFrame): The Polars DataFrame.
        gpkg_file_name (str): The GeoPackage file name.
        layer_name (str): The layer name.
    """
    list_columns: list[str] = [
        name for name, col_type in dict(table.schema).items() if type(col_type) == pl.List]
    table_pd: pd.DataFrame = table.with_columns(
        c(list_columns).list.join(", ")
    ).to_pandas()

    table_pd["geometry"] = table_pd["geometry"].apply(from_wkt)
    table_pd = table_pd[table_pd.geometry.notnull()]
    table_gpd: gpd.GeoDataFrame = gpd.GeoDataFrame(
        table_pd.dropna(axis=0, subset="geometry"), crs=settings.SWISS_SRID) # type: ignore
    table_gpd = table_gpd[~table_gpd["geometry"].is_empty] # type: ignore
    table_gpd.to_file(gpkg_file_name, layer=layer_name) 


def dict_to_gpkg(data: dict, file_path: str):
    """
    Save a dictionary of Polars DataFrames as a GeoPackage file.

    Args:
        data (dict): The dictionary of Polars DataFrames.
        file_path (str): The GeoPackage file path.
    """
    with tqdm.tqdm(range(1), ncols=100, desc="Save input data in gpkg format") as pbar:
        for layer_name, table in data.items():
            if isinstance(table, pl.DataFrame):
                if not table.is_empty():
                    table_to_gpkg(table=table, gpkg_file_name=file_path, layer_name=layer_name)
        pbar.update()

def dict_to_duckdb(data: dict[str, pl.DataFrame], file_path: str):
    """
    Save a dictionary of Polars DataFrames as a DuckDB file.

    Args:
        data (dict[str, pl.DataFrame]): The dictionary of Polars DataFrames.
        file_path (str): The DuckDB file path.
    """
    build_non_existing_dirs(os.path.dirname(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)
    with duckdb.connect(file_path) as con:
        con.execute("SET TimeZone='UTC'")
        for table_name, table_pl in tqdm.tqdm(data.items(), desc="Save dictionary into duckdb file", ncols=150):
            query = f"CREATE TABLE {table_name} AS SELECT * FROM table_pl"
            con.execute(query)
                
                
def duckdb_to_dict(file_path: str) -> dict:
    """
    Load a DuckDB file into a dictionary of Polars DataFrames.

    Args:
        file_path (str): The DuckDB file path.

    Returns:
        dict: The dictionary of Polars DataFrames.
    """
    schema_dict: dict[str, pl.DataFrame] = {} # type: ignore

    with duckdb.connect(database=file_path) as con:
        con.execute("SET TimeZone='UTC'")
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        pbar = tqdm.tqdm(
            con.execute(query).fetchall(), ncols=150, 
            desc="Read and validate tables from {} file".format(os.path.basename(file_path))
            )
        for table_name in pbar:
            query: str = f"SELECT * FROM {table_name[0]}"
            schema_dict[table_name[0]] = con.execute(query).pl()
                    
    return schema_dict


# def filter_unique_nodes_from_list(node_id_list: pl.Expr)-> pl.Expr:
    
#     return (
#         pl.when(node_id_list.list.len() == 1)
#         .then(node_id_list.list.get(0, null_on_oob=True))
#         .otherwise(pl.lit(None))
#     )


def dictionary_key_filtering(dictionary: dict, key_list: list) -> dict:
    """
    Filter a dictionary by a list of keys.

    Args:
        dictionary (dict): The dictionary to filter.
        key_list (list): The list of keys to keep.

    Returns:
        dict: The filtered dictionary.
    """
    return dict(filter(lambda x : x[0] in key_list, dictionary.items()))


def generate_uuid(base_value: str, base_uuid: uuid.UUID | None = None, added_string: str = "") -> str:
    """
    Generate a UUID based on a base value, base UUID, and an optional added string.

    Args:
        base_value (str): The base value for generating the UUID.
        base_uuid (uuid.UUID, optional): The base UUID for generating the UUID.
        added_string (str, optional): The optional added string. Defaults to "".

    Returns:
        str: The generated UUID.
    """
    if base_uuid is None:
        base_uuid=NAMESPACE_UUID
    return str(uuid.uuid5(base_uuid, added_string + base_value))



