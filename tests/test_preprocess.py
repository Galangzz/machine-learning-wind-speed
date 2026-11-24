import numpy as np
import pandas as pd
import pytest

from scripts.preprocess import aggregate_daily, aggregate_monthly, parse_midc


def raw_row(year=2023, doy=1, mst=0, ws=5.0, wd=180.0, temp=10.0, rh=50.0, pres=805.0, precip=0.0):
    return {
        "Year": year,
        "DOY": doy,
        "MST": mst,
        "Avg Wind Speed @ 50m [m/s]": ws,
        "Avg Wind Direction @ 50m [deg]": wd,
        "Temperature @ 50m [deg C]": temp,
        "Relative Humidity [%]": rh,
        "Station Pressure [mBar]": pres,
        "Precipitation (Accumulated) [mm]": precip,
    }


def minute_day(date: str, **overrides) -> pd.DataFrame:
    idx = pd.date_range(date, periods=1440, freq="min")
    df = pd.DataFrame(index=idx)
    df["wind_speed_50m"] = 5.0
    df["wind_dir_50m"] = 180.0
    df["temperature_50m"] = 10.0
    df["relative_humidity"] = 50.0
    df["pressure"] = 805.0
    df["precipitation"] = 0.0
    for column, value in overrides.items():
        df[column] = value
    return df


def test_parse_midc_treats_mst_as_hhmm():
    raw = pd.DataFrame([raw_row(mst=0), raw_row(mst=59), raw_row(mst=100), raw_row(mst=2359)])
    out = parse_midc(raw)
    assert list(out.index) == [
        pd.Timestamp("2023-01-01 00:00"),
        pd.Timestamp("2023-01-01 00:59"),
        pd.Timestamp("2023-01-01 01:00"),
        pd.Timestamp("2023-01-01 23:59"),
    ]


def test_parse_midc_maps_sentinels_and_360_degrees():
    raw = pd.DataFrame([
        raw_row(ws=-9999, wd=-7999, temp=-6999, rh=-999, pres=-9999, precip=-6999),
        raw_row(mst=1, wd=360.0),
    ])
    out = parse_midc(raw)
    assert out.iloc[0].isna().all()
    assert out.loc[pd.Timestamp("2023-01-01 00:01"), "wind_dir_50m"] == 0.0


def test_daily_precipitation_ignores_midnight_stale_value():
    df = minute_day("2023-01-02")
    df.iloc[0, df.columns.get_loc("precipitation")] = 5.588
    df.iloc[1:6, df.columns.get_loc("precipitation")] = 0.0
    df.iloc[720, df.columns.get_loc("precipitation")] = 1.778
    out = aggregate_daily(df)
    assert out.loc["2023-01-02", "precipitation"] == pytest.approx(1.778)


def test_daily_wind_direction_uses_vector_mean():
    df = minute_day("2023-01-02")
    df.iloc[:720, df.columns.get_loc("wind_dir_50m")] = 350.0
    df.iloc[720:, df.columns.get_loc("wind_dir_50m")] = 10.0
    out = aggregate_daily(df)
    deg = out.loc["2023-01-02", "wind_dir_deg"]
    assert min(abs(deg), abs(deg - 360.0)) < 1.0


def test_daily_completeness_threshold_is_75_percent():
    df = pd.concat([minute_day("2023-01-02"), minute_day("2023-01-03")])
    df.iloc[:1079, df.columns.get_loc("wind_speed_50m")] = 5.0
    df.iloc[1079:1440, df.columns.get_loc("wind_speed_50m")] = np.nan
    df.iloc[1440:1440 + 1080, df.columns.get_loc("wind_speed_50m")] = 5.0
    df.iloc[1440 + 1080:, df.columns.get_loc("wind_speed_50m")] = np.nan
    out = aggregate_daily(df)
    assert np.isnan(out.loc["2023-01-02", "wind_speed_50m"])
    assert out.loc["2023-01-03", "wind_speed_50m"] == pytest.approx(5.0)


def test_monthly_mean_uses_all_minutes():
    df = pd.concat([minute_day("2023-01-01"), minute_day("2023-01-02")])
    df.iloc[:1440, df.columns.get_loc("wind_speed_50m")] = 10.0
    df.iloc[1440:1440 + 1080, df.columns.get_loc("wind_speed_50m")] = 0.0
    df.iloc[1440 + 1080:, df.columns.get_loc("wind_speed_50m")] = np.nan
    out = aggregate_monthly(df, min_completeness=0.0)
    expected = (1440 * 10.0 + 1080 * 0.0) / (1440 + 1080)
    assert out.loc[pd.Period("2023-01", "M"), "wind_speed_50m"] == pytest.approx(expected)


def test_parse_midc_maps_midc_invalid_marker():
    raw = pd.DataFrame([
        raw_row(ws=-99999, wd=-99999, temp=-99999, rh=-99999, pres=-99999, precip=-99999),
    ])
    out = parse_midc(raw)
    assert out.iloc[0].isna().all()


def test_parse_midc_masks_physically_impossible_values():
    raw = pd.DataFrame([
        raw_row(temp=-266.0),
        raw_row(mst=1, ws=-0.5),
        raw_row(mst=2, rh=150.0),
        raw_row(mst=3, pres=50.0),
        raw_row(mst=4, precip=-1.0),
        raw_row(mst=5),
    ])
    out = parse_midc(raw)
    assert np.isnan(out.loc[pd.Timestamp("2023-01-01 00:00"), "temperature_50m"])
    assert np.isnan(out.loc[pd.Timestamp("2023-01-01 00:01"), "wind_speed_50m"])
    assert np.isnan(out.loc[pd.Timestamp("2023-01-01 00:02"), "relative_humidity"])
    assert np.isnan(out.loc[pd.Timestamp("2023-01-01 00:03"), "pressure"])
    assert np.isnan(out.loc[pd.Timestamp("2023-01-01 00:04"), "precipitation"])
    assert out.loc[pd.Timestamp("2023-01-01 00:05"), "wind_speed_50m"] == pytest.approx(5.0)
