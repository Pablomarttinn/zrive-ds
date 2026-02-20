import requests
import pandas as pd
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
    df_variables = pd.DataFrame(data["daily"])
    df_variables["time"] = pd.to_datetime(df_variables["time"])

    return df_variables


def calculo_estadisticos(df):
    df["time"] = df["time"].dt.to_period("M")
    df_wind = df.groupby("time")[["wind_speed_10m_max"]].mean()
    df_precipitation = df.groupby("time")[["precipitation_sum"]].sum()
    df_temp = df.groupby("time")[["temperature_2m_mean"]].mean()

    return df_wind, df_precipitation, df_temp


def graficar(tuple_df, ciudad):
    fig, axes = plt.subplots(nrows=len(tuple_df), ncols=1, figsize=(8, 5))

    for ax, df in zip(axes, tuple_df):
        df.index = df.index.to_timestamp()
        var = df.columns[0]
        ax.plot(df.index, df[var], label="Monthly Mean")
        ax.set_title(f"{var} in {ciudad}")

    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Mes")
    fig.tight_layout()
    plt.show()


def main():
    mad_data = get_data_meteo_api("Madrid")
    df_mad_variables = extraccion_variables(mad_data)
    df_mad_statistics = calculo_estadisticos(df_mad_variables)
    graficar(df_mad_statistics, "Madrid")

    lon_data = get_data_meteo_api("London")
    df_lon_variables = extraccion_variables(lon_data)
    df_lon_statistics = calculo_estadisticos(df_lon_variables)
    graficar(df_lon_statistics, "London")

    rio_data = get_data_meteo_api("Rio")
    df_rio_variables = extraccion_variables(rio_data)
    df_rio_statistics = calculo_estadisticos(df_rio_variables)
    graficar(df_rio_statistics, "Rio")


if __name__ == "__main__":
    main()
