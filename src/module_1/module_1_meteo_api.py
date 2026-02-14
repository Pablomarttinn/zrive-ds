import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from jsonschema import validate, ValidationError

api_url = "https://archive-api.open-meteo.com/v1/archive"

coordinates = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

schema_meteo: dict[str, dict] = {
    "type": "object",
    "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "daily_units": {"type": "object"},
        "daily": {
            "type": "object",
            "properties": {
                "time": {"type": "array", "items": {"type": "string"}},
                "temperature_2m_mean": {"type": "array", "items": {"type": "number"}},
                "precipitation_sum": {"type": "array", "items": {"type": "number"}},
                "wind_speed_10m_max": {"type": "array", "items": {"type": "number"}},
            },
            "required": [
                "time",
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
            ],
        },
    },
    "required": ["daily", "daily_units"],
}


def call_api(url, params, schema):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            validate(instance=data, schema=schema)
            print("Answer meets schema")
            return data
        except ValidationError as e:
            print("Answer does not meet schema", e)
            return None
    else:
        print("Error:", response.status_code)
        return None


def get_data_meteo_api(ciudad):
    try:
        params = {"start_date": "2010-01-01", "end_date": "2020-12-31"}
        params["latitude"] = coordinates[ciudad]["latitude"]
        params["longitude"] = coordinates[ciudad]["longitude"]
        params["daily"] = ",".join(variables)
    except KeyError:
        print(f"{ciudad} is not defined in Coordinates, please define and try again")
        return None

    data = call_api(api_url, params, schema_meteo)

    if data is None:
        print("Unable to obtain data")
        return None
    else:
        return data


def extraccion_variables(data):
    variables = {}
    variables["time"] = [
        datetime.strptime(date, "%Y-%m-%d") for date in data["daily"]["time"]
    ]
    variables["units"] = data["daily_units"].copy()
    for key in data["daily"].keys():
        if key != "time":
            variables[key] = np.array(data["daily"][key])

    return variables


def graficar(variables, ciudad):
    x = variables["time"][::21]
    variable_keys = [k for k in variables.keys() if k != "time" and k != "units"]
    for key in variable_keys:
        plt.figure()
        plt.plot(x, variables[key][::21])
        plt.title(f"{key} in {ciudad} from 2010 to 2020")
        plt.xlabel("time")
        plt.ylabel(f"{key} in {variables['units'][key]}")
        plt.show(block=False)
    input("Press Enter to close the figures")


if __name__ == "__main__":
    mad_data = get_data_meteo_api("Madrid")
    mad_interest_data = extraccion_variables(mad_data)
    graficar(mad_interest_data, "Madrid")

    lon_data = get_data_meteo_api("London")
    lon_interest_data = extraccion_variables(lon_data)
    graficar(lon_interest_data, "London")

    rio_data = get_data_meteo_api("Rio")
    rio_interest_data = extraccion_variables(rio_data)
    graficar(rio_interest_data, "Rio")
