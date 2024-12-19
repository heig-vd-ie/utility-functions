import unittest
import polars as pl
import uuid
from datetime import datetime

from polars_function import (
    generate_uuid_col, cast_float, cast_boolean, modify_string_col, parse_date, 
    parse_timestamp, cast_to_utc_timestamp, generate_random_uuid, get_meta_data_string, digitize_col
)

class TestPolarsFunctions(unittest.TestCase):

    def test_generate_uuid_col(self):
        df = pl.DataFrame({"col": ["a", "b", "c"]})
        result = df.with_columns(generate_uuid_col(pl.col("col")).alias("uuid_col"))
        self.assertEqual(result["uuid_col"].n_unique(), 3)

    def test_cast_float(self):
        df = pl.DataFrame({"col": ["1,23", "4,56", "7,89"]})
        result = df.with_columns(cast_float(pl.col("col")).alias("float_col"))
        self.assertTrue(result["float_col"].dtype == pl.Float64)

    def test_cast_boolean(self):
        df = pl.DataFrame({"col": ["true", "false", "oui", "non"]})
        result = df.with_columns(cast_boolean(pl.col("col")).alias("bool_col"))
        self.assertTrue(result["bool_col"].dtype == pl.Boolean)

    def test_modify_string_col(self):
        df = pl.DataFrame({"col": ["a-b", "c-d", "e-f"]})
        format_str = {"-": "_"}
        result = df.with_columns(modify_string_col(pl.col("col"), format_str).alias("modified_col"))
        self.assertEqual(result["modified_col"].to_list(), ["a_b", "c_d", "e_f"])

    def test_parse_date(self):
        date_str = "2023-10-05"
        default_date = datetime(2020, 1, 1)
        result = parse_date(date_str, default_date)
        self.assertEqual(result, datetime(2023, 10, 5))

    def test_parse_timestamp(self):
        df = pl.DataFrame({"col": ["2023-10-05 12:34:56"]})
        result = df.with_columns(parse_timestamp(pl.col("col"), "2023-10-05 12:34:56").alias("timestamp_col"))
        self.assertTrue(result["timestamp_col"].dtype == pl.Datetime)

    def test_cast_to_utc_timestamp(self):
        df = pl.DataFrame({"col": [datetime(2023, 10, 5, 12, 34, 56)]})
        result = df.with_columns(cast_to_utc_timestamp(pl.col("col")).alias("utc_col"))
        self.assertTrue(result["utc_col"].dtype == pl.Datetime)

    def test_generate_random_uuid(self):
        df = pl.DataFrame({"col": ["a", "b", "c"]})
        result = df.with_columns(generate_random_uuid(pl.col("col")).alias("uuid_col"))
        self.assertEqual(result["uuid_col"].n_unique(), 3)

    def test_get_meta_data_string(self):
        df = pl.DataFrame({"col": [{"key1": "value1", "key2": None}, {"key1": "value2", "key2": "value3"}]})
        result = df.with_columns(get_meta_data_string(pl.col("col")).alias("json_col"))
        self.assertEqual(result["json_col"].to_list(), ['{"key1": "value1"}', '{"key1": "value2", "key2": "value3"}'])

    def test_digitize_col(self):
        df = pl.DataFrame({"col": [1.0, 2.5, 3.0, 4.5, 5.0]})
        result = df.with_columns(digitize_col(pl.col("col"), 1.0, 5.0, 4).alias("digitized_col"))
        self.assertTrue(result["digitized_col"].dtype == pl.Int64)

if __name__ == "__main__":
    unittest.main()
