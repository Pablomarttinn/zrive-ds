from src.module_1.module_1_meteo_api import (
    call_api,
    get_data_meteo_api,
    extraccion_variables,
    calculo_estadisticos,
)
from unittest.mock import patch, Mock
import pandas as pd


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
    data = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "wind_speed_10m_max": [5.0, 7.0],
            "precipitation_sum": [0.0, 2.5],
            "temperature_2m_mean": [10.0, 12.0],
        }
    }

    df = extraccion_variables(data)

    # Type
    assert isinstance(df, pd.DataFrame)

    # Expected columns
    expected_cols = {
        "time",
        "wind_speed_10m_max",
        "precipitation_sum",
        "temperature_2m_mean",
    }
    assert expected_cols.issubset(df.columns)

    # Type of time column
    assert pd.api.types.is_datetime64_any_dtype(df["time"])


def test_calculo_estadisticos():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "wind_speed_10m_max": [4.0, 6.0, 10.0],
            "precipitation_sum": [1.0, 2.0, 3.0],
            "temperature_2m_mean": [8.0, 12.0, 20.0],
        }
    )

    df_wind, df_precip, df_temp = calculo_estadisticos(df)

    # Types
    assert isinstance(df_wind, pd.DataFrame)
    assert isinstance(df_precip, pd.DataFrame)
    assert isinstance(df_temp, pd.DataFrame)

    # Monthly index
    assert isinstance(df_wind.index, pd.PeriodIndex)
    assert df_wind.index.freqstr == "M"

    # Only the row we want
    assert df_wind.shape[1] == 1
    assert df_precip.shape[1] == 1
    assert df_temp.shape[1] == 1

    # Expected values
    # Enero wind mean = (4 + 6) / 2 = 5
    assert df_wind.loc["2024-01", "wind_speed_10m_max"] == 5.0

    # Febrero precip sum = 3
    assert df_precip.loc["2024-02", "precipitation_sum"] == 3.0

    # Febrero temp mean = 20
    assert df_temp.loc["2024-02", "temperature_2m_mean"] == 20.0
