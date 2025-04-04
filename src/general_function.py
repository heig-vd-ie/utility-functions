"""
Auxiliary functions
"""
import logging
import os
import uuid
import coloredlogs
import polars as pl
import logging
from polars import col as c
from typing import Optional, Union
import zipfile
import tarfile
import re 
import owncloud
import tqdm
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkt

NAMESPACE_UUID: uuid.UUID = uuid.UUID('{bc4d4e0c-98c9-11ec-b909-0242ac120002}')
SWISS_SRID: int = 2056


def initialize_output_file(file_path: str):
    """
    Initialize an output file by creating necessary directories and removing the file if it already exists.

    Args:
        file_path (str): The path of the file to initialize.
    """
    build_non_existing_dirs(file_path=os.path.dirname(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)

def extract_archive(file_name: str, extracted_folder: Optional[str] = None, force_extraction: bool = False) -> None:
    """
    Extract an archive file to a specified folder.

    Args:
        file_name (str): The name of the archive file.
        extracted_folder (Optional[str], optional): The folder to extract the files to. Defaults to None.
        force_extraction (bool, optional): Whether to force extraction even if the folder already exists. Defaults to False.
    """
    
    if extracted_folder is None:
        extracted_folder, extension = os.path.splitext(file_name)
    else:
        extension = os.path.splitext(file_name)[1]
    
    if not force_extraction and os.path.exists(extracted_folder):
        return
    if extension == ".tar":
        file = tarfile.open(file_name, "r")
        with tqdm.tqdm(total=1, desc=f"Extract {file_name} archive") as pbar:
            file.extractall(extracted_folder, filter="data") # type: ignore
            pbar.update(1)
    elif extension == ".tgz":
        file = tarfile.open(file_name, "r:gz")
        with tqdm.tqdm(total=1, desc=f"Extract {file_name} archive") as pbar:
            file.extractall(extracted_folder, filter="data") # type: ignore
            pbar.update(1)
    elif extension == ".zip":
        file = zipfile.ZipFile(file_name, "r")
        with tqdm.tqdm(total=1, desc=f"Extract {file_name} archive") as pbar:
            file.extractall(extracted_folder) # type: ignore
            pbar.update(1)
    else:
        raise ValueError(f"{extension} format not supported")
    

def scan_folder(
    folder_name: str, extension: Optional[Union[str, list[str]]] = None, file_names: Optional[str] = None) -> list[str]:
    """
    Scan a folder and return a list of file paths with specified extensions or names.

    Args:
        folder_name (str): The folder to scan.
        extension (Optional[Union[str, list[str]]], optional): The file extensions to filter by. Defaults to None.
        file_names (Optional[str], optional): The file names to filter by. Defaults to None.

    Returns:
        list[str]: List of file paths.
    """
    file_list: list = []
    if isinstance(extension, str):
        extension = [extension]
    for entry in list(os.scandir(folder_name)):
        if entry.is_dir():
            file_list.extend(scan_folder(folder_name=entry.path, extension=extension, file_names=file_names))
        file_path = entry.path
        file_ext = os.path.splitext(file_path)[1]
        if extension is None:
            if file_names is None:
                file_list.append(file_path)
            elif file_names in file_path:
                file_list.append(file_path)
        elif file_ext in extension:
            if file_names is None:
                file_list.append(file_path)
            elif file_names in file_path:
                file_list.append(file_path)  
    return file_list


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
        if "_trash" not in file_data.name:
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

def pl_to_dict_with_tuple(df: pl.DataFrame) -> dict:
    """
    Convert a Polars DataFrame with two columns into a dictionary where the first column
    contains tuples as keys and the second column contains the values.

    Args:
        df (pl.DataFrame): Polars DataFrame with two columns.

    Returns:
        dict: Dictionary representation of the DataFrame with tuples as keys.

    Raises:
        ValueError: If the DataFrame does not have exactly two columns.

    Example:
    >>> import polars as pl
    >>> data = {'key': [[1, 2], [3, 4], [5, 6]], 'value': [10, 20, 30]}
    >>> df = pl.DataFrame(data)
    >>> pl_to_dict_with_tuple(df)
    {(1, 2): 10, (3, 4): 20, (5, 6): 30}
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame is not composed of two columns")
    return dict(map(
        lambda data: (tuple(data[0]), data[1]), df.rows()
    ))


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

def table_to_gpkg(table: pl.DataFrame, gpkg_file_name: str, layer_name: str, srid: int = SWISS_SRID):
    """
    Save a Polars DataFrame as a GeoPackage file. As GeoPackage does not support list columns, 
    the list columns are joined into a single string separated with a comma.

    Args:
        table (pl.DataFrame): The Polars DataFrame.
        gpkg_file_name (str): The GeoPackage file name.
        layer_name (str): The layer name.
        srid (int, optional): The SRID. Defaults to SWISS_SRID.
    """
    list_columns: list[str] = [
        name for name, col_type in dict(table.schema).items() if type(col_type) == pl.List]
    table_pd: pd.DataFrame = table.with_columns(
        c(list_columns).cast(pl.List(pl.Utf8)).list.join(", ")
    ).to_pandas()

    table_pd["geometry"] = table_pd["geometry"].apply(from_wkt)
    table_pd = table_pd[table_pd.geometry.notnull()]
    table_gpd: gpd.GeoDataFrame = gpd.GeoDataFrame(
        table_pd.dropna(axis=0, subset="geometry"), crs=srid) # type: ignore
    table_gpd = table_gpd[~table_gpd["geometry"].is_empty] # type: ignore
    # Save gpkg without logging
    logger = logging.getLogger("pyogrio")
    previous_level = logger.level 
    logger.setLevel(logging.WARNING)
    table_gpd.to_file(gpkg_file_name, layer=layer_name) 
    logger.setLevel(previous_level)  
    

def dict_to_gpkg(data: dict, file_path: str, srid: int = SWISS_SRID):
    """
    Save a dictionary of Polars DataFrames as a GeoPackage file.

    Args:
        data (dict): The dictionary of Polars DataFrames.
        file_path (str): The GeoPackage file path.
        srid (int, optional): The SRID. Defaults to SWISS_SRID.
    """
    with tqdm.tqdm(range(1), ncols=100, desc="Save input data in gpkg format") as pbar:
        for layer_name, table in data.items():
            if isinstance(table, pl.DataFrame):
                if not table.is_empty():
                    table_to_gpkg(table=table, gpkg_file_name=file_path, layer_name=layer_name, srid=srid)
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
        pbar = tqdm.tqdm(data.items(), ncols=150, desc="Save dictionary into duckdb file")
        for table_name, table_pl in pbar:
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



