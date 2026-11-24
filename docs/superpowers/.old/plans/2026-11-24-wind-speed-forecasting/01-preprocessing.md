# Plan 01 — Preprocessing Core + Regression Tests

> **Plan 01 of 09.** Baca `00-overview.md` dan `AGENTS.md` dulu. Selesaikan plan ini sebelum Plan 02.

**Goal:** Membuat modul preprocessing MIDC yang menangani jebakan data (MST HHMM, sentinel, presipitasi akumulatif, arah angin sirkular, completeness) dan mengunci perilakunya dengan regression test.

**Depends on:** —

**Files:**
- Create: `scripts/preprocess.py`
- Create: `tests/test_preprocess.py`
- Verify/modify: `.gitignore`
- Modify: `pyproject.toml`, `uv.lock` (via `uv add --dev pytest`)

**Interfaces:**
- Consumes: raw MIDC DataFrame from `pd.read_csv`.
- Produces:
  - `COLUMN_MAP: dict[str, str]`
  - `SENTINELS: tuple[int, ...]`
  - `parse_midc(df: pd.DataFrame) -> pd.DataFrame`
  - `aggregate_daily(df: pd.DataFrame, min_completeness: float = 0.75) -> pd.DataFrame`
  - `aggregate_monthly(df: pd.DataFrame, min_completeness: float = 0.75) -> pd.DataFrame`

**Output contracts:**
- `parse_midc`: `DatetimeIndex` bernama `datetime`; kolom persis `wind_speed_50m`, `wind_dir_50m`, `temperature_50m`, `relative_humidity`, `pressure`, `precipitation`.
- `aggregate_daily`: `DatetimeIndex` bernama `date`; kolom `wind_speed_50m`, `wind_dir_deg`, `wind_dir_sin`, `wind_dir_cos`, `temperature_50m`, `relative_humidity`, `pressure`, `precipitation`, `n_valid_minutes`, `completeness`.
- `aggregate_monthly`: `PeriodIndex` freq `M` bernama `month`; kolom sama seperti daily.

**Constraints khusus task ini:**
- `MST` = HHMM (`hour = MST // 100`, `minute = MST % 100`).
- Sentinel `-9999, -7999, -6999, -999` → `NaN`.
- Completeness `>= 0.75` dari 1.440 menit.
- Presipitasi: buang 00:00, lalu `max` harian; jangan `sum`.
- Vector mean arah angin: `atan2(mean(sin), mean(cos))`; 360° = 0°.

---

- [ ] **Step 1: Verify `.gitignore`**

Pastikan `.gitignore` memuat:

```gitignore
data/raw/
.ipynb_checkpoints/
```

Jika belum ada, tambahkan. File ini kemungkinan sudah di-commit; tidak perlu commit ulang kalau tidak berubah.

- [ ] **Step 2: Add pytest as a dev dependency**

Run: `uv add --dev pytest`
Expected: `pyproject.toml` dan `uv.lock` berubah, pytest terpasang.

- [ ] **Step 3: Write the failing tests**

Buat `tests/test_preprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'scripts.preprocess'`.

- [ ] **Step 5: Implement `scripts/preprocess.py`**

Konstanta:

```python
SENTINELS = (-9999, -7999, -6999, -999)

COLUMN_MAP = {
    "Avg Wind Speed @ 50m [m/s]": "wind_speed_50m",
    "Avg Wind Direction @ 50m [deg]": "wind_dir_50m",
    "Temperature @ 50m [deg C]": "temperature_50m",
    "Relative Humidity [%]": "relative_humidity",
    "Station Pressure [mBar]": "pressure",
    "Precipitation (Accumulated) [mm]": "precipitation",
}
```

`parse_midc`:
- Copy DataFrame, strip nama kolom, validasi kolom wajib.
- `mst = df["MST"].astype(int)`; validasi `0 <= mst <= 2359` dan `(mst % 100) < 60`.
- Bangun timestamp dari `Year` + `DOY` (`pd.to_datetime(..., format="%Y-%j")`) + `mst // 100` jam + `mst % 100` menit.
- Rename kolom via `COLUMN_MAP`, coerce ke numeric, mask sentinel menjadi `NaN`.
- `wind_dir_50m = wind_dir_50m % 360.0` (lakukan setelah masking).
- Set `DatetimeIndex`, sort, dan raise `ValueError` jika ada duplikat timestamp.

`aggregate_daily`:
- Group by `df.index.normalize()`.
- Mean untuk `wind_speed_50m`, `temperature_50m`, `relative_humidity`, `pressure`.
- Vector mean untuk arah angin: `x = mean(sin(rad))`, `y = mean(cos(rad))`, `deg = degrees(arctan2(x, y)) % 360`, lalu `sin`/`cos` dari `deg`. **Perhatikan urutan: `atan2(sin, cos)`, bukan `atan2(cos, sin)`.**
- Presipitasi: `midnight = (index.hour == 0) & (index.minute == 0)`; group `df.loc[~midnight, "precipitation"]` by day, ambil `max`.
- `counts = df.groupby(day).count()`; `n_valid_minutes = counts["wind_speed_50m"]`; `completeness = n_valid_minutes / 1440.0`.
- Masking: untuk `wind_speed_50m`, `temperature_50m`, `relative_humidity`, `pressure`, `precipitation`, set `NaN` jika `counts[col] / 1440 < min_completeness`. Untuk arah angin, mask `wind_dir_deg`, `wind_dir_sin`, `wind_dir_cos` jika `counts["wind_dir_50m"] / 1440 < min_completeness`.

`aggregate_monthly`:
- Group by `df.index.to_period("M")`.
- Mean/vector mean sama seperti daily, tetapi `expected_minutes = period.days_in_month * 1440`.
- Presipitasi bulanan = jumlah dari daily max (buang 00:00) per bulan.
- Masking memakai `counts / expected_minutes < min_completeness`.
- Index `PeriodIndex` freq `M` bernama `month`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock scripts/preprocess.py tests/test_preprocess.py
git commit -m "feat: add tested MIDC preprocessing"
```

---

**Verification gate:** `uv run pytest tests/test_preprocess.py -v` → 6 passed. Jangan lanjut ke Plan 02 sebelum ini lulus.
