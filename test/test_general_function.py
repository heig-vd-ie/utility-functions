import unittest
import uuid
import polars as pl
from datetime import datetime
from general_function import (
    generate_log, pl_to_dict, modify_string, camel_to_snake, snake_to_camel,
    convert_list_to_string, generate_uuid
)

class TestGeneralFunctions(unittest.TestCase):

    def test_generate_log(self):
        log = generate_log(name="test_log")
        self.assertEqual(log.name, "test_log")

    def test_pl_to_dict(self):
        df = pl.DataFrame({"key": ["a", "b", "c"], "value": [1, 2, 3]})
        result = pl_to_dict(df)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_modify_string(self):
        string = "a-b-c"
        format_str = {"-": "_"}
        result = modify_string(string, format_str)
        self.assertEqual(result, "a_b_c")

    def test_camel_to_snake(self):
        string = "camelCaseString"
        result = camel_to_snake(string)
        self.assertEqual(result, "camel_case_string")

    def test_snake_to_camel(self):
        string = "snake_case_string"
        result = snake_to_camel(string)
        self.assertEqual(result, "SnakeCaseString")

    def test_convert_list_to_string(self):
        list_data = [1, 2, 3]
        result = convert_list_to_string(list_data)
        self.assertEqual(result, "1, 2, 3")

    def test_generate_uuid(self):
        base_value = "test_value"
        result = generate_uuid(base_value)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 36)

if __name__ == "__main__":
    unittest.main()
