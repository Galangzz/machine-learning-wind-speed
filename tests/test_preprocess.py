import numpy as np
import pandas as pd
import pytest

from scripts.preprocess import (
    aggregate_daily,
    aggregate_monthly,
    fill_feature_gaps,
    parse_midc,
)


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


def test_parse_midc_rejects_duplicate_timestamps():
    raw = pd.DataFrame([raw_row(mst=100), raw_row(mst=100)])
    with pytest.raises(ValueError, match="duplikat"):
        parse_midc(raw)


def test_monthly_precipitation_sums_daily_maxima_not_minutes():
    df = pd.concat([minute_day("2023-01-02"), minute_day("2023-01-03")])
    precip = df.columns.get_loc("precipitation")
    df.iloc[0, precip] = 5.588  # stale value carried from the previous day
    df.iloc[1, precip] = 1.0
    df.iloc[1440, precip] = 7.0  # stale again
    df.iloc[1441 + 720, precip] = 2.5
    out = aggregate_monthly(df, min_completeness=0.0)
    assert out.loc[pd.Period("2023-01", "M"), "precipitation"] == pytest.approx(3.5)


def test_monthly_completeness_masks_sparse_months():
    out = aggregate_monthly(minute_day("2023-01-02"))
    month = pd.Period("2023-01", "M")
    assert np.isnan(out.loc[month, "wind_speed_50m"])
    assert out.loc[month, "n_valid_minutes"] == 1440
    assert out.loc[month, "completeness"] == pytest.approx(1440 / (31 * 1440))


def test_fill_feature_gaps_never_uses_future_values():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame({"temperature_50m": [1.0, np.nan, 100.0, 4.0]}, index=idx)
    filled = fill_feature_gaps(df, ["temperature_50m"])
    assert filled.loc[idx[1], "temperature_50m"] == pytest.approx(1.0)


def test_fill_feature_gaps_respects_limit_and_leading_gaps():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame({"pressure": [np.nan, 1.0, np.nan, np.nan, np.nan, 2.0]}, index=idx)
    filled = fill_feature_gaps(df, ["pressure"], limit=2)
    assert np.isnan(filled.loc[idx[0], "pressure"])
    assert filled.loc[idx[2], "pressure"] == pytest.approx(1.0)
    assert filled.loc[idx[3], "pressure"] == pytest.approx(1.0)
    assert np.isnan(filled.loc[idx[4], "pressure"])
    assert filled.loc[idx[5], "pressure"] == pytest.approx(2.0)


def test_fill_feature_gaps_leaves_unlisted_columns_untouched():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"pressure": [1.0, np.nan, 3.0], "wind_speed_50m": [5.0, np.nan, 6.0]}, index=idx
    )
    filled = fill_feature_gaps(df, ["pressure"])
    assert filled.loc[idx[1], "pressure"] == pytest.approx(1.0)
    assert np.isnan(filled.loc[idx[1], "wind_speed_50m"])
