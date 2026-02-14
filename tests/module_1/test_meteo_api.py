
from src.module_1.module_1_meteo_api import (
    call_api,
    get_data_meteo_api,
    extraccion_variables,
)
from unittest.mock import patch, Mock
from datetime import datetime
import numpy as np


def test_success_call_api():
    fake_response = Mock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"name": "Pablo"}

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    with patch(
        "src.module_1.module_1_meteo_api.requests.get", return_value=fake_response
    ):
        result = call_api("http://fake", {}, schema)

    assert result == {"name": "Pablo"}


def test_SchemaFail_call_api():
    fake_response = Mock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"name": 123}

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    with patch(
        "src.module_1.module_1_meteo_api.requests.get", return_value=fake_response
    ):
        result = call_api("http//:fake", {}, schema)

    assert result is None


def test_httpFail_call_api():
    fake_response = Mock()
    fake_response.status_code = 400
    fake_response.json.return_value = {"name": "Pablo"}

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    with patch(
        "src.module_1.module_1_meteo_api.requests.get", return_value=fake_response
    ):
        result = call_api("http//:fake", {}, schema)

    assert result is None


def test_get_data_meteo_api_fail():
    with patch("src.module_1.module_1_meteo_api.call_api", return_value=None):
        result = get_data_meteo_api("Madrid")

    assert result is None


def test_get_data_meteo_api_success():
    fake_reponse = "Pablo"
    with patch("src.module_1.module_1_meteo_api.call_api", return_value=fake_reponse):
        result = get_data_meteo_api("Madrid")

    assert result == "Pablo"


def test_get_data_meteo_api_cityFail():
    ciudad = "Japon"
    result = get_data_meteo_api(ciudad)

    assert result is None


def test_extraccion_variables():
    fake_data = {
        "daily_units": {"Time": "x", "Var1": "y", "Var2": "z"},
        "daily": {
            "time": ["2010-02-14", "2015-03-12"],
            "Var1": [1, 2, 3, 4],
            "Var2": [1, 3, 4, 5],
        },
    }
    fecha_str = ["2010-02-14", "2015-03-12"]
    fecha_dt = [datetime.strptime(date, "%Y-%m-%d") for date in fecha_str]
    variables = {
        "time": fecha_dt,
        "units": {"Time": "x", "Var1": "y", "Var2": "z"},
        "Var1": np.array([1, 2, 3, 4]),
        "Var2": np.array([1, 3, 4, 5]),
    }
    result = extraccion_variables(fake_data)

    assert result["time"] == variables["time"]
    assert result["units"] == variables["units"]
    np.testing.assert_array_equal(result["Var1"], variables["Var1"])
    np.testing.assert_array_equal(result["Var2"], variables["Var2"])
