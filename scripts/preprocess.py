"""Parsing dan agregasi data MIDC NWTC M2 (resolusi 1 menit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTINELS = (-99999, -9999, -7999, -6999, -999)

# Batas fisik longgar untuk situs NWTC M2; nilai di luar ini adalah glitch sensor.
PHYSICAL_BOUNDS = {
    "wind_speed_50m": (0.0, 75.0),
    "wind_dir_50m": (0.0, 360.0),
    "temperature_50m": (-90.0, 60.0),
    "relative_humidity": (0.0, 100.0),
    "pressure": (300.0, 1100.0),
    "precipitation": (0.0, 500.0),
}

COLUMN_MAP = {
    "Avg Wind Speed @ 50m [m/s]": "wind_speed_50m",
    "Avg Wind Direction @ 50m [deg]": "wind_dir_50m",
    "Temperature @ 50m [deg C]": "temperature_50m",
    "Relative Humidity [%]": "relative_humidity",
    "Station Pressure [mBar]": "pressure",
    "Precipitation (Accumulated) [mm]": "precipitation",
}

MEAN_COLUMNS = ("wind_speed_50m", "temperature_50m", "relative_humidity", "pressure")

OUTPUT_COLUMNS = [
    "wind_speed_50m",
    "wind_dir_deg",
    "wind_dir_sin",
    "wind_dir_cos",
    "temperature_50m",
    "relative_humidity",
    "pressure",
    "precipitation",
    "n_valid_minutes",
    "completeness",
]

MINUTES_PER_DAY = 1440


def parse_midc(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah export CSV MIDC menjadi frame 1 menit ber-index `datetime`.

    `MST` adalah HHMM (`100` berarti 01:00), bukan jumlah menit sejak tengah malam.
    """
    out = df.copy()
    out.columns = [str(column).strip() for column in out.columns]

    required = ["Year", "DOY", "MST", *COLUMN_MAP]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"kolom wajib tidak ditemukan: {missing}")

    year = pd.to_numeric(out["Year"], errors="coerce")
    doy = pd.to_numeric(out["DOY"], errors="coerce")
    mst = pd.to_numeric(out["MST"], errors="coerce")
    if year.isna().any() or doy.isna().any() or mst.isna().any():
        raise ValueError("Year, DOY, dan MST harus numerik dan tidak boleh kosong")

    mst = mst.astype(int)
    if ((mst < 0) | (mst > 2359)).any():
        raise ValueError("MST berada di luar rentang 0-2359")
    if ((mst % 100) >= 60).any():
        raise ValueError("MST memiliki menit >= 60; MST harus berformat HHMM")

    day = pd.to_datetime(
        year.astype(int).astype(str) + "-" + doy.astype(int).astype(str).str.zfill(3),
        format="%Y-%j",
    )
    stamps = (
        day
        + pd.to_timedelta(mst // 100, unit="h")
        + pd.to_timedelta(mst % 100, unit="m")
    )

    out = out.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.mask(out.isin(SENTINELS))
    for column, (low, high) in PHYSICAL_BOUNDS.items():
        out[column] = out[column].where(out[column].between(low, high))
    out["wind_dir_50m"] = out["wind_dir_50m"] % 360.0

    out.index = pd.DatetimeIndex(stamps, name="datetime")
    out = out.sort_index()

    if out.index.duplicated().any():
        duplicated = out.index[out.index.duplicated()][:3].tolist()
        raise ValueError(f"timestamp duplikat setelah parsing MST sebagai HHMM: {duplicated}")

    return out


def _vector_mean_deg(direction: pd.Series, keys) -> pd.Series:
    radians = np.deg2rad(direction)
    sin_mean = np.sin(radians).groupby(keys).mean()
    cos_mean = np.cos(radians).groupby(keys).mean()
    mean_deg = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0
    return pd.Series(mean_deg, index=sin_mean.index, dtype="float64")


def _daily_precipitation(df: pd.DataFrame) -> pd.Series:
    midnight = (df.index.hour == 0) & (df.index.minute == 0)
    accumulated = df.loc[~midnight, "precipitation"]
    return accumulated.groupby(accumulated.index.normalize()).max()


def aggregate_daily(df: pd.DataFrame, min_completeness: float = 0.75) -> pd.DataFrame:
    """Agregat harian dari data 1 menit; hari di bawah ambang completeness menjadi NaN."""
    days = df.index.normalize()
    counts = df.groupby(days).count()
    fractions = counts / MINUTES_PER_DAY

    out = pd.DataFrame(index=counts.index)
    out.index.name = "date"

    for column in MEAN_COLUMNS:
        out[column] = df.groupby(days)[column].mean().where(fractions[column] >= min_completeness)

    direction_ok = fractions["wind_dir_50m"] >= min_completeness
    out["wind_dir_deg"] = _vector_mean_deg(df["wind_dir_50m"], days).where(direction_ok)
    out["wind_dir_sin"] = np.sin(np.deg2rad(out["wind_dir_deg"]))
    out["wind_dir_cos"] = np.cos(np.deg2rad(out["wind_dir_deg"]))

    out["precipitation"] = _daily_precipitation(df).reindex(out.index).where(
        fractions["precipitation"] >= min_completeness
    )

    out["n_valid_minutes"] = counts["wind_speed_50m"].astype(int)
    out["completeness"] = counts["wind_speed_50m"] / MINUTES_PER_DAY

    return out[OUTPUT_COLUMNS]


def aggregate_monthly(df: pd.DataFrame, min_completeness: float = 0.75) -> pd.DataFrame:
    """Agregat bulanan langsung dari data 1 menit, bukan dari rata-rata harian."""
    months = df.index.to_period("M")
    counts = df.groupby(months).count()
    expected_minutes = pd.Series(
        [period.days_in_month * MINUTES_PER_DAY for period in counts.index],
        index=counts.index,
        dtype="float64",
    )
    fractions = counts.div(expected_minutes, axis=0)

    out = pd.DataFrame(index=counts.index)
    out.index.name = "month"

    for column in MEAN_COLUMNS:
        out[column] = df.groupby(months)[column].mean().where(fractions[column] >= min_completeness)

    direction_ok = fractions["wind_dir_50m"] >= min_completeness
    out["wind_dir_deg"] = _vector_mean_deg(df["wind_dir_50m"], months).where(direction_ok)
    out["wind_dir_sin"] = np.sin(np.deg2rad(out["wind_dir_deg"]))
    out["wind_dir_cos"] = np.cos(np.deg2rad(out["wind_dir_deg"]))

    daily_precip = _daily_precipitation(df)
    monthly_precip = daily_precip.groupby(daily_precip.index.to_period("M")).sum()
    out["precipitation"] = monthly_precip.reindex(out.index).where(
        fractions["precipitation"] >= min_completeness
    )

    out["n_valid_minutes"] = counts["wind_speed_50m"].astype(int)
    out["completeness"] = counts["wind_speed_50m"] / expected_minutes

    return out[OUTPUT_COLUMNS]
