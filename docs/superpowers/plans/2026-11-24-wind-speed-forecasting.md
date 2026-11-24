# Daily t+1 Wind Speed Forecasting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun notebook Jupyter yang reproducible dan laporan Markdown berbahasa Indonesia yang mereplikasi Altork (2025) sebagai sanity check, lalu mengevaluasi peramalan kecepatan angin harian satu hari ke depan (t+1) di NWTC M2.

**Architecture:** Downloader MIDC yang checkpointed mengubah export CSV 1 menit per bulan menjadi agregat harian dan bulanan. Notebook memuat agregat tersebut, menjalankan sanity check bulanan, membangun dataset supervised t+1, melakukan tuning dan evaluasi lima predictor, lalu menyimpan semua metrik/gambar ke `results/`. Laporan hanya mengutip angka dari `results/`.

**Tech Stack:** Python 3.14 (uv), pandas 3.1, numpy 2.5, scikit-learn 1.9, matplotlib, seaborn, Jupyter.

**Spec:** `docs/superpowers/specs/2026-11-24-wind-speed-forecasting-design.md`

**Target internal:** 28 November 2026. **Deadline:** 1 Desember 2026. Hari 29 Nov–1 Des menjadi buffer.

---

## Global Constraints

- Sumber data: `https://midcdmz.nlr.gov/apps/data_api.pl?site=NWTC&begin=YYYYMMDD&end=YYYYMMDD`; site `NWTC`; periode 2019-01-01–2023-12-31.
- Kolom sumber dan nama pendek:
  - `Avg Wind Speed @ 50m [m/s]` → `wind_speed_50m`
  - `Avg Wind Direction @ 50m [deg]` → `wind_dir_50m`
  - `Temperature @ 50m [deg C]` → `temperature_50m`
  - `Relative Humidity [%]` → `relative_humidity`
  - `Station Pressure [mBar]` → `pressure`
  - `Precipitation (Accumulated) [mm]` → `precipitation`
- `MST` adalah HHMM: `hour = MST // 100`, `minute = MST % 100`. Bukan menit sejak tengah malam.
- Sentinel `-9999, -7999, -6999, -999` → `NaN`.
- Completeness harian: `>= 0.75` dari 1.440 menit valid (1.080 menit). Di bawah itu agregat variabel = `NaN`.
- Presipitasi: buang menit 00:00, lalu `max` per hari. Jangan `sum`.
- Arah angin: vector mean, lalu encode sin/cos. 360° = 0°.
- Agregat bulanan dihitung langsung dari 1 menit, bukan dari rata-rata harian.
- Split eksperimen utama berdasarkan `target_date`: train ≤ 2021-12-31, validation 2022, test 2023. Tidak ada shuffling. **Pengecualian:** sanity check replikasi memakai random split 70/15/15 dengan `random_state=42`, sesuai paper.
- Scaler fit pada train saja; saat refit final, fit pada train+validation saja.
- `random_state=42` di semua split, model, dan permutation importance.
- Model: persistence, Linear Regression, SVR RBF, Fine Tree (`DecisionTreeRegressor`), Random Forest. Tidak ada LSTM/GRU/Transformer/XGBoost/stacking.
- Metrik: RMSE, MSE, MAE, R². RMSE dan MAE reduction terhadap persistence wajib dihitung.
- Laporan `report.md` berbahasa Indonesia. Semua angka berasal dari `results/*.csv`.
- Jangan commit `data/raw/`, `.venv/`, `.ipynb_checkpoints/`.

## Review Focus

Lima kegagalan yang paling mungkin terjadi dan harus ditutup oleh test/audit:

1. **`MST` dibaca sebagai menit** → timestamp duplikat, tanggal bergeser, dan file Januari 2023 tidak lagi 44.640 baris. Test: `test_parse_midc_treats_mst_as_hhmm`.
2. **Presipitasi dijumlahkan atau nilai 00:00 ikut dihitung** → total harian ribuan mm, atau hari kering setelah hujan mendapat nilai hujan hari sebelumnya. Test: `test_daily_precipitation_ignores_midnight_stale_value`.
3. **Arah angin dirata-ratakan linear** → 350° dan 10° menjadi 180°, bukan 0°. Test: `test_daily_wind_direction_uses_vector_mean`.
4. **Hari dengan data sedikit tetap dianggap lengkap** → satu jam data mendapat bobot sama dengan satu hari penuh. Test: `test_daily_completeness_threshold_is_75_percent`.
5. **Leakage** → random split, scaler fit pada seluruh data, atau meteorologi hari t+1 masuk fitur. Audit di Task 5 memakai assertion eksplisit.

---

## Verified Data Facts (dipakai sebagai fixture dan ekspektasi)

Sudah diverifikasi langsung pada 24 November 2026:

- Januari 2023: 44.640 baris data, tepat 31 hari × 1.440 menit, tanpa duplikat setelah `MST` di-parse sebagai HHMM. Rentang: 2023-01-01 00:00–2023-01-31 23:59.
- Februari 2020 (tahun kabisat): 41.760 baris, tepat 29 hari × 1.440 menit. Rentang: 2020-02-01 00:00–2020-02-29 23:59.
- Enam kolom modeling tidak memiliki sentinel atau NA pada kedua sampel tersebut, tetapi kolom lain memilikinya; parser tetap wajib menangani sentinel.
- Presipitasi adalah akumulasi harian yang reset pada 00:01. Contoh stale value:
  - 2023-01-03 00:00 = 2,286 mm (sisa 2 Januari); hujan aktual 3 Januari = 0.
  - 2023-01-19 00:00 = 6,096 mm (sisa 18 Januari); hujan aktual 19 Januari = 0.
  - 2020-02-04 00:00 = 5,588 mm (sisa 3 Februari); hujan aktual 4 Februari = 1,778 mm.
- API: satu hari ±8 detik; satu bulan ±27 detik dan 18–22 MB. Full 60 bulan ±27 menit. Jangan query setahun penuh sekaligus (pernah timeout dan menghasilkan file terpotong).

---

## Task 1: Preprocessing Core + Regression Tests

**Files:**
- Create: `scripts/preprocess.py`
- Create: `tests/test_preprocess.py`
- Modify: `.gitignore`
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

- [ ] **Step 1: Update `.gitignore`**

Tambahkan:

```gitignore
data/raw/
.ipynb_checkpoints/
```

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

Implementasi wajib mengikuti aturan berikut:

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
- Vector mean untuk arah angin: `x = mean(sin(rad))`, `y = mean(cos(rad))`, `deg = degrees(arctan2(x, y)) % 360`, lalu `sin`/`cos` dari `deg`. (Perhatikan urutan: `atan2(sin, cos)`, bukan `atan2(cos, sin)`.)
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
git add .gitignore pyproject.toml uv.lock scripts/preprocess.py tests/test_preprocess.py
git commit -m "feat: add tested MIDC preprocessing"
```

---

## Task 2: Download and Aggregate 2019–2023

**Files:**
- Create: `scripts/download_data.py`
- Generated: `data/processed/daily.csv`, `data/processed/monthly.csv`, `data/processed/data_quality.csv`
- Generated (gitignored): `data/raw/nwtc_YYYY_MM.csv`

**Interfaces:**
- Consumes: `parse_midc`, `aggregate_daily`, `aggregate_monthly`.
- Produces: `fetch_month(year: int, month: int, raw_dir: Path, retries: int = 3) -> Path`, `main(start=(2019, 1), end=(2023, 12)) -> None`.

- [ ] **Step 1: Write `scripts/download_data.py`**

Wajib memiliki:
- `BASE_URL = "https://midcdmz.nlr.gov/apps/data_api.pl"`.
- `fetch_month`: hitung `start` = hari pertama bulan, `end` = akhir bulan; request dengan `params={"site": "NWTC", "begin": ..., "end": ...}`, `timeout=180`.
- Resume: jika file sudah ada dan jumlah barisnya `>= 99%` dari `days_in_month * 1440`, lewati download.
- Retry 3× dengan backoff 5/10 detik; tulis ke `.part` lalu `replace()` agar atomic.
- Validasi respons: status 2xx dan jumlah baris `>= 99%` dari ekspektasi. Jangan menerima file terpotong.
- `main`: loop 60 bulan, parse tiap bulan, catat quality metrics, agregasi, gabungkan.
- Setelah concat: `daily = daily.reindex(pd.date_range("2019-01-01", "2023-12-31", freq="D"))`; `daily.index.name = "date"`.
- `monthly = monthly.reindex(pd.period_range("2019-01", "2023-12", freq="M"))`; `monthly.index.name = "month"`.
- Simpan CSV dengan `float_format="%.6f"`.
- `data_quality.csv` minimal berisi: `year, month, expected_minutes, n_rows, n_missing_minutes, valid_wind_speed, valid_wind_dir, valid_temperature, valid_humidity, valid_pressure, valid_precipitation`.

- [ ] **Step 2: Smoke test one month**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import pandas as pd
from scripts.download_data import fetch_month
from scripts.preprocess import parse_midc

p = fetch_month(2023, 1, Path("data/raw"))
df = parse_midc(pd.read_csv(p))
print("rows:", len(df), "| range:", df.index.min(), "->", df.index.max())
assert len(df) == 44640
assert df.index.min() == pd.Timestamp("2023-01-01 00:00")
assert df.index.max() == pd.Timestamp("2023-01-31 23:59")
assert not df.index.duplicated().any()
PY
```

Expected: `rows: 44640 | range: 2023-01-01 00:00:00 -> 2023-01-31 23:59:00`, tanpa assertion error.

- [ ] **Step 3: Run the full download**

Run: `uv run python -m scripts.download_data`
Expected: 60 bulan diproses; ±27 menit. Jika dijalankan di background, tunggu sampai selesai sebelum lanjut. Jika satu bulan gagal, rerun akan resume dari file yang sudah lengkap. Gunakan `-m` (bukan `python scripts/download_data.py`) agar import `scripts.preprocess` bekerja.

- [ ] **Step 4: Verify the processed datasets**

Run:

```bash
uv run python - <<'PY'
import pandas as pd

daily = pd.read_csv("data/processed/daily.csv", parse_dates=["date"], index_col="date")
monthly = pd.read_csv("data/processed/monthly.csv", index_col="month")
monthly.index = pd.PeriodIndex(monthly.index, freq="M")
quality = pd.read_csv("data/processed/data_quality.csv")

print("daily:", daily.shape, daily.index.min(), "->", daily.index.max())
print("monthly:", monthly.shape, monthly.index.min(), "->", monthly.index.max())
print("quality rows:", len(quality))
print("missing wind_speed days:", int(daily["wind_speed_50m"].isna().sum()))
print("duplicate dates:", int(daily.index.duplicated().sum()))

assert len(daily) == 1826
assert len(monthly) == 60
assert len(quality) == 60
assert daily.index.min() == pd.Timestamp("2019-01-01")
assert daily.index.max() == pd.Timestamp("2023-12-31")
assert monthly.index.min() == pd.Period("2019-01", "M")
assert monthly.index.max() == pd.Period("2023-12", "M")
assert not daily.index.duplicated().any()
PY
```

Expected: `daily: (1826, ...)`, `monthly: (60, ...)`, `quality rows: 60`, `duplicate dates: 0`. Jumlah `missing wind_speed days` dilaporkan apa adanya (target: 0, tetapi bisa >0 jika ada gap).

- [ ] **Step 5: Commit**

```bash
git add scripts/download_data.py data/processed/daily.csv data/processed/monthly.csv data/processed/data_quality.csv
git commit -m "feat: download and aggregate NWTC M2 2019-2023"
```

---

## Task 3: Notebook Skeleton, EDA, and Data Quality

**Files:**
- Modify: `main.ipynb`
- Create (generated): `results/figures/`

**Interfaces:**
- Consumes: `data/processed/daily.csv`, `monthly.csv`, `data_quality.csv`.
- Produces: notebook sections 1–3, figure files.

**Notebook conventions:** satu cell untuk satu langkah logis; semua cell harus jalan berurutan; tidak ada state tersembunyi; setiap figure disimpan ke `results/figures/` dengan `dpi=150, bbox_inches="tight"`.

- [ ] **Step 1: Replace `main.ipynb` with the project notebook**

Struktur Markdown yang wajib ada:

```markdown
# Daily One-Day-Ahead Wind Speed Forecasting — NWTC M2
## 1. Setup
## 2. Data Loading and Quality
## 3. Exploratory Data Analysis
```

Cell section 1:
- Import `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `pathlib.Path`.
- Import scikit-learn: `clone`, `RandomForestRegressor`, `permutation_importance`, `LinearRegression`, `mean_absolute_error`, `mean_squared_error`, `r2_score`, `GridSearchCV`, `KFold`, `TimeSeriesSplit`, `train_test_split`, `make_pipeline`, `MinMaxScaler`, `SVR`, `DecisionTreeRegressor`.
- Konstanta: `DATA_DIR = Path("data/processed")`, `FIG_DIR = Path("results/figures")`, `RESULTS_DIR = Path("results")`, `RANDOM_STATE = 42`.
- `FIG_DIR.mkdir(parents=True, exist_ok=True)`, `RESULTS_DIR.mkdir(parents=True, exist_ok=True)`.
- `pd.set_option("display.max_columns", 100)`, `sns.set_theme(style="whitegrid")`.

Cell section 2:
- Load ketiga CSV.
- `monthly.index = pd.PeriodIndex(monthly.index, freq="M")` setelah `read_csv`.
- Tampilkan `daily.head()`, `monthly.head()`, `quality.head()`.
- Assertions persis seperti Task 2 Step 4.
- Tabel ringkasan: jumlah hari, rentang tanggal, missing per kolom, jumlah bulan, missing per bulan.
- Hitung `valid_wind_speed / expected_minutes` dari `data_quality.csv` dan tampilkan 5 bulan terendah.

Cell section 3:
- Time series harian `wind_speed_50m` 2019–2023 → `fig_daily_wind_speed.png`.
- Time series bulanan `wind_speed_50m` 2019–2023 → `fig_monthly_wind_speed.png`.
- Histogram + KDE `wind_speed_50m` harian → `fig_target_distribution.png`.
- Rata-rata bulanan per bulan kalender (Jan–Dec) → `fig_monthly_climatology.png`.
- Correlation heatmap fitur harian (`wind_speed_50m`, `wind_dir_sin`, `wind_dir_cos`, `temperature_50m`, `relative_humidity`, `pressure`, `precipitation`) → `fig_correlation.png`.
- Tabel korelasi Pearson dengan target, diurutkan.

- [ ] **Step 2: Execute the notebook headless**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0, tidak ada traceback.

- [ ] **Step 3: Verify figures and outputs**

Run: `ls -lh results/figures/`
Expected: minimal 5 file PNG yang disebut di Step 1.

- [ ] **Step 4: Commit**

```bash
git add main.ipynb results/figures
git commit -m "feat: add notebook EDA and data quality report"
```

---

## Task 4: Monthly Sanity Check (Replication)

**Files:**
- Modify: `main.ipynb`
- Create: `results/sanity_check_metrics.csv`

**Interfaces:**
- Consumes: `monthly.csv` (60 baris).
- Produces: `results/sanity_check_metrics.csv` dengan kolom `model, RMSE, MSE, MAE, R2`.

- [ ] **Step 1: Add `## 4. Monthly Sanity Check (Replication)` to the notebook**

Cell 4.1 — dataset:
- Fitur: `["pressure", "temperature_50m", "wind_dir_deg", "relative_humidity", "precipitation"]` (lima fitur, arah angin satu kolom derajat).
- Target: `wind_speed_50m`.
- `data = monthly.dropna(subset=features + [target])`; cetak jumlah observasi (target 60).
- Split 70/15/15 acak: `train_test_split(..., test_size=0.30, random_state=42)`, lalu split sisa 50/50 menjadi validation dan test. Cetak ukuran tiap split (target 42/9/9).
- `MinMaxScaler(feature_range=(-1, 1))` fit pada train saja.

Cell 4.2 — models:
- Linear Regression default.
- SVR RBF dengan grid kecil: `C ∈ {1, 10, 100}`, `gamma ∈ {"scale", 0.1}`, `epsilon ∈ {0.01, 0.1}`; pilih berdasarkan validation RMSE (boleh `GridSearchCV` dengan `KFold(3, shuffle=True, random_state=42)` pada train, lalu evaluasi validation).
- Fine Tree: `DecisionTreeRegressor(random_state=42)` dengan grid `max_depth ∈ {None, 3, 5, 10}`, `min_samples_leaf ∈ {1, 2, 4}`.
- Untuk setiap model: fit pada train, evaluasi validation, refit pada train+validation dengan scaler baru, evaluasi test.
- Hitung RMSE, MSE, MAE, R² dengan `mean_squared_error`, `mean_absolute_error`, `r2_score`.

Cell 4.3 — reporting:
- Tabel metrik test ketiga model, diurutkan berdasarkan RMSE.
- Tabel pembanding berisi angka paper: LR RMSE 0,29555 / MSE 0,087352 / MAE 0,18061 / R² 0,97; SVM RMSE 0,32275 / R² 0,96; Fine Tree RMSE 0,44042 / R² 0,93.
- Catatan eksplisit: angka tidak harus identik karena paper tidak merinci hyperparameter, split, dan agregasi.
- Sanity guard: semua metrik finite; R² Linear Regression harus `> 0.5` dan `< 0.999`. Jika di luar rentang, selidiki leakage atau parsing sebelum lanjut.
- Simpan `results/sanity_check_metrics.csv`.

- [ ] **Step 2: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb`
Expected: exit code 0; `results/sanity_check_metrics.csv` berisi 3 baris.

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/sanity_check_metrics.csv
git commit -m "feat: add monthly replication sanity check"
```

---

## Task 5: t+1 Supervised Dataset, Persistence, and Linear Regression

**Files:**
- Modify: `main.ipynb`
- Create: `results/test_metrics.csv`

**Interfaces:**
- Consumes: `daily.csv`.
- Produces:
  - `FEATURES` (16 nama, urutan tetap).
  - `train`, `val`, `test` DataFrames.
  - `results/test_metrics.csv` dengan kolom `model, RMSE, MSE, MAE, R2, RMSE_reduction_pct, MAE_reduction_pct`.

- [ ] **Step 1: Add `## 5. t+1 Dataset and Linear Baseline`**

Cell 5.1 — feature set (copy persis):

```python
FEATURES = [
    "wind_speed_lag0", "wind_speed_lag1", "wind_speed_lag2", "wind_speed_lag3",
    "wind_speed_lag4", "wind_speed_lag5", "wind_speed_lag6", "wind_speed_lag7",
    "wind_dir_sin", "wind_dir_cos", "temperature_50m", "pressure",
    "relative_humidity", "precipitation", "doy_sin", "doy_cos",
]
```

Cell 5.2 — interpolate short feature gaps, then build supervised frame:

Interpolasi hanya untuk fitur non-wind-speed. **Jangan interpolasi `wind_speed_50m` atau target.**

```python
FEATURE_COLS_TO_INTERPOLATE = [
    "wind_dir_sin", "wind_dir_cos", "temperature_50m",
    "relative_humidity", "pressure", "precipitation",
]
missing_before = daily[FEATURE_COLS_TO_INTERPOLATE].isna().sum()
daily[FEATURE_COLS_TO_INTERPOLATE] = daily[FEATURE_COLS_TO_INTERPOLATE].interpolate(
    method="linear", limit=2, limit_area="inside"
)
missing_after = daily[FEATURE_COLS_TO_INTERPOLATE].isna().sum()
print(pd.DataFrame({"before": missing_before, "after": missing_after}))

sup = daily.sort_index().copy()
for lag in range(8):
    sup[f"wind_speed_lag{lag}"] = sup["wind_speed_50m"].shift(lag)
sup["target"] = sup["wind_speed_50m"].shift(-1)
sup["target_date"] = sup.index + pd.Timedelta(days=1)
doy = sup["target_date"].dt.dayofyear
sup["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
sup["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
sup = sup.dropna(subset=FEATURES + ["target"]).copy()
print("supervised rows after drop:", len(sup), "of", len(daily))
```

Cell 5.3 — leakage audit (wajib, jangan dilewati):

```python
assert (sup["target_date"] == sup.index + pd.Timedelta(days=1)).all()
assert np.allclose(
    sup["target"].to_numpy(),
    daily.reindex(sup["target_date"])["wind_speed_50m"].to_numpy(),
)
assert np.allclose(
    sup["wind_speed_lag0"].to_numpy(),
    daily.reindex(sup.index)["wind_speed_50m"].to_numpy(),
)
assert set(FEATURES).isdisjoint({"target", "target_date"})
```

Cell 5.4 — chronological split by target date:

```python
train = sup[sup["target_date"] <= pd.Timestamp("2021-12-31")]
val = sup[(sup["target_date"] >= pd.Timestamp("2022-01-01")) & (sup["target_date"] <= pd.Timestamp("2022-12-31"))]
test = sup[sup["target_date"] >= pd.Timestamp("2023-01-01")]

assert len(train) + len(val) + len(test) == len(sup)
assert train["target_date"].max() < val["target_date"].min()
assert val["target_date"].max() < test["target_date"].min()
```

Cetak `len(train), len(val), len(test)` dan rentang `target_date` masing-masing. Target: train ±1.088, validation 365, test 365 (berkurang jika ada hari hilang; 7 hari pertama dan 1 hari terakhir selalu hilang karena lag/target).

Cell 5.5 — metrics helper:

```python
def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

results = []

def add_result(model_name, y_true, y_pred):
    row = {"model": model_name, **regression_metrics(y_true, y_pred)}
    results.append(row)
    return row
```

Cell 5.6 — persistence:

```python
add_result("Persistence", test["target"], test["wind_speed_lag0"])
persistence_rmse = results[0]["RMSE"]
persistence_mae = results[0]["MAE"]
```

Cell 5.7 — Linear Regression:
- `lr_scaler = MinMaxScaler(feature_range=(-1, 1))` fit pada `train[FEATURES]`.
- Audit scaling (wajib): `assert np.allclose(lr_scaler.data_min_, train[FEATURES].min().to_numpy())` dan `assert np.allclose(lr_scaler.data_max_, train[FEATURES].max().to_numpy())`.
- Fit `LinearRegression()` pada train yang sudah di-scale (untuk memastikan alur train-only jalan).
- Buat final model sebagai pipeline: `lr_final = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), LinearRegression())`; fit pada `pd.concat([train, val])[FEATURES]` dan `pd.concat([train, val])["target"]`.
- Evaluasi `lr_final` pada test; `add_result("Linear Regression", test["target"], pred)`.
- Simpan `final_models = {"Linear Regression": lr_final}` untuk Task 6.

Cell 5.8 — finalize table:
- Hitung `RMSE_reduction_pct` dan `MAE_reduction_pct` terhadap persistence.
- Simpan `pd.DataFrame(results).sort_values("RMSE")` ke `results/test_metrics.csv` (index=False).
- Tampilkan tabel.

- [ ] **Step 2: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb`
Expected: exit code 0; `results/test_metrics.csv` berisi 2 baris (Persistence, Linear Regression).

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/test_metrics.csv
git commit -m "feat: add t+1 dataset and persistence/linear baselines"
```

---

## Task 6: SVR, Fine Tree, and Random Forest with Tuning

**Files:**
- Modify: `main.ipynb`
- Create: `results/best_params.json`
- Update: `results/test_metrics.csv`

**Interfaces:**
- Consumes: `train`, `val`, `test`, `FEATURES` from Task 5.
- Produces: final metrics for SVR, Fine Tree, RF; `results/best_params.json`.

- [ ] **Step 1: Add `## 6. Nonlinear Models and Tuning`**

Cell 6.1 — shared setup:

```python
tscv = TimeSeriesSplit(n_splits=3)
X_train, y_train = train[FEATURES], train["target"]
X_val, y_val = val[FEATURES], val["target"]
X_test, y_test = test[FEATURES], test["target"]
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
```

Cell 6.2 — SVR:

```python
svr_pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), SVR(kernel="rbf"))
svr_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.1],
    "svr__epsilon": [0.01, 0.1, 0.5],
}
svr_search = GridSearchCV(svr_pipe, svr_grid, cv=tscv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
svr_search.fit(X_train, y_train)
```

Cell 6.3 — Fine Tree:

```python
tree_grid = {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 4, 8]}
tree_search = GridSearchCV(DecisionTreeRegressor(random_state=42), tree_grid, cv=tscv,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
tree_search.fit(X_train, y_train)
```

Cell 6.4 — Random Forest:

```python
rf_grid = {
    "n_estimators": [300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0],
}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1,
)
rf_search.fit(X_train, y_train)
```

Cell 6.5 — validation + refit + test untuk setiap model:
- Evaluasi `best_estimator_` pada validation; catat `val_rmse`.
- `clone(best_estimator_)`, fit pada `X_trainval, y_trainval`.
- Prediksi test; `add_result`.
- Tambahkan model final ke dictionary `final_models` yang sudah berisi `"Linear Regression"` dari Task 5: `final_models.update({"SVR": ..., "Fine Tree": ..., "Random Forest": ...})`.

Cell 6.6 — best params + final table:
- Simpan `{"SVR": {"params": svr_search.best_params_, "val_rmse": ...}, ...}` ke `results/best_params.json` dengan indent 2.
- Hitung ulang reduction vs persistence, tulis `results/test_metrics.csv` (5 baris), tampilkan diurutkan berdasarkan RMSE.
- Sanity guard: semua metrik finite; RF RMSE harus `< 100` (tidak meledak).

- [ ] **Step 2: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb`
Expected: exit code 0; `results/test_metrics.csv` berisi 5 baris; `results/best_params.json` ada.

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/test_metrics.csv results/best_params.json
git commit -m "feat: tune and evaluate SVR, fine tree, and random forest"
```

---

## Task 7: Feature Importance and Error Analysis

**Files:**
- Modify: `main.ipynb`
- Create: `results/feature_importance.csv`, `results/error_by_season.csv`, `results/error_by_level.csv`
- Create figures in `results/figures/`

**Interfaces:**
- Consumes: `final_models`, `test`, `FEATURES`.
- Produces: three CSV tables and four figures.

- [ ] **Step 1: Add `## 7. Feature Importance and Error Analysis`**

Cell 7.1 — importance table:
- `rf = final_models["Random Forest"]`; impurity importance `rf.feature_importances_`.
- Permutation importance pada test untuk model terbaik (RMSE terendah): `permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring="neg_root_mean_squared_error", n_jobs=-1)`.
- Koefisien Linear Regression: `final_models["Linear Regression"].named_steps["linearregression"].coef_`; catat bahwa interpretasi koefisien hanya valid setelah scaling.
- Gabungkan ke DataFrame `feature, rf_impurity, permutation_mean, permutation_std, lr_coef`; urutkan berdasarkan `permutation_mean` menurun; simpan `results/feature_importance.csv`.
- Plot top-10 permutation importance → `fig_feature_importance.png`.

Cell 7.2 — error by season:
- Mapping: DJF = 12,1,2; MAM = 3,4,5; JJA = 6,7,8; SON = 9,10,11 berdasarkan `target_date.dt.month`.
- Untuk model terbaik dan persistence, hitung RMSE dan MAE per musim.
- Simpan `results/error_by_season.csv` dengan kolom `season, model, RMSE, MAE, n`.
- Plot grouped bar chart → `fig_error_by_season.png`.

Cell 7.3 — error by wind speed level:
- `pd.qcut(test["target"], 3, labels=["low", "mid", "high"], duplicates="drop")`.
- Hitung RMSE/MAE per level untuk model terbaik dan persistence.
- Simpan `results/error_by_level.csv`.
- Plot → `fig_error_by_level.png`.

Cell 7.4 — prediction plots:
- Time series actual vs model terbaik vs persistence pada test 2023 → `fig_test_timeseries.png`.
- Scatter predicted vs actual model terbaik dengan garis 1:1 → `fig_test_scatter.png`.
- Histogram residual model terbaik → `fig_residuals.png`.

Cell 7.5 — ringkasan:
- Tampilkan tabel akhir `results/test_metrics.csv` diurutkan berdasarkan RMSE.
- Cetak model terbaik dan RMSE reduction terhadap persistence.
- Cetak top-5 fitur berdasarkan permutation importance.
- Cetak satu paragraf singkat: apakah model ML mengalahkan persistence, dan fitur apa yang dominan.

- [ ] **Step 2: Execute the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb`
Expected: exit code 0; tiga CSV dan enam PNG baru ada (importance, seasonal error, level error, time series, scatter, residuals).

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/feature_importance.csv results/error_by_season.csv results/error_by_level.csv results/figures
git commit -m "feat: add feature importance and error analysis"
```

---

## Task 8: Indonesian Markdown Report

**Files:**
- Create: `report.md`

**Interfaces:**
- Consumes: semua `results/*.csv`, `results/best_params.json`, `results/figures/*.png`, dan output notebook.
- Produces: `report.md` berbahasa Indonesia.

- [ ] **Step 1: Write `report.md` with this structure**

```markdown
# Peramalan Kecepatan Angin Harian Satu Hari ke Depan: Replikasi dan Ekstensi Perbandingan Model Machine Learning

## Abstrak
## 1. Pendahuluan
## 2. Tinjauan Pustaka
## 3. Data dan Preprocessing
## 4. Metodologi
## 5. Hasil Sanity Check Replikasi
## 6. Hasil Peramalan t+1
## 7. Diskusi
## 8. Keterbatasan
## 9. Kesimpulan
## Referensi
## Lampiran
```

Konten wajib:
- **Abstrak:** 150–250 kata; sebutkan tujuan, data, model, temuan utama.
- **Pendahuluan:** latar belakang energi angin, paper Altork (2025), gap same-period vs forecasting, pertanyaan riset.
- **Tinjauan Pustaka:** tiga model paper, Random Forest (Breiman 2001), data sirkular (Fisher 1993), peramalan time series (Hyndman & Athanasopoulos 2021).
- **Data dan Preprocessing:** sumber MIDC/NWTC M2, periode 2019–2023, agregasi 1 menit → harian/bulanan, aturan completeness 75%, penanganan presipitasi dan arah angin, missing data.
- **Metodologi:** definisi t+1, 16 fitur, split kronologis, scaling, hyperparameter tuning, metrik, persistence baseline.
- **Hasil Sanity Check:** tabel 3 model + angka paper; jelaskan bahwa angka tidak harus identik.
- **Hasil t+1:** tabel 5 model (persistence + 4 model) dengan RMSE/MSE/MAE/R² dan reduction vs persistence; feature importance; error per musim dan level; gambar.
- **Diskusi:** mengapa R² turun dibanding paper; apakah model non-linear mengungguli LR; apakah ML mengalahkan persistence; fitur dominan; implikasi praktis.
- **Keterbatasan:** satu lokasi, 5 tahun, resolusi harian, tanpa NWP, tanpa interval ketidakpastian, presipitasi jarang, asumsi stationarity.
- **Kesimpulan:** jawab pertanyaan riset secara eksplisit.
- **Referensi:** Altork (2025), Jager & Andreas (1996), Breiman (2001), Fisher (1993), Hyndman & Athanasopoulos (2021).
- **Lampiran:** hyperparameter final dari `best_params.json`, versi library, seed.

Aturan:
- Setiap tabel angka harus bisa dilacak ke CSV di `results/`.
- Gunakan gambar dengan path relatif, contoh `![Feature importance](results/figures/fig_feature_importance.png)`.
- Jangan mengklaim paper salah; posisikan sebagai ekstensi dan koreksi protokol.
- Jangan menulis angka yang tidak ada di `results/`.

- [ ] **Step 2: Consistency check**

Run:

```bash
uv run python - <<'PY'
import json
import pandas as pd

for path in [
    "results/sanity_check_metrics.csv",
    "results/test_metrics.csv",
    "results/feature_importance.csv",
    "results/error_by_season.csv",
    "results/error_by_level.csv",
]:
    df = pd.read_csv(path)
    print(path, df.shape)
    assert not df.empty

with open("results/best_params.json") as f:
    print(json.load(f))
PY
```

Expected: semua file ada dan tidak kosong.

- [ ] **Step 3: Commit**

```bash
git add report.md
git commit -m "docs: add Indonesian analysis report"
```

---

## Task 9: Reproducibility and Final Verification

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md` (optional: short run instructions)
- Modify: `main.ipynb` (final executed outputs)

- [ ] **Step 1: Pin the environment**

Run:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project > requirements.txt
```

Expected: `requirements.txt` berisi versi pinned (`==`) untuk seluruh dependency.

- [ ] **Step 2: Run tests**

Run: `uv run pytest -q`
Expected: 6 passed.

- [ ] **Step 3: Execute the notebook from a clean kernel**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0, tidak ada traceback. Notebook tersimpan dengan outputs.

- [ ] **Step 4: Verify the Definition of Done**

Run:

```bash
uv run python - <<'PY'
import pandas as pd

daily = pd.read_csv("data/processed/daily.csv")
monthly = pd.read_csv("data/processed/monthly.csv")
main = pd.read_csv("results/test_metrics.csv")
sanity = pd.read_csv("results/sanity_check_metrics.csv")

print("daily rows:", len(daily))
print("monthly rows:", len(monthly))
print("main models:", main["model"].tolist())
print("sanity models:", sanity["model"].tolist())

assert len(daily) == 1826
assert len(monthly) == 60
assert set(main["model"]) == {"Persistence", "Linear Regression", "SVR", "Fine Tree", "Random Forest"}
assert set(sanity["model"]) == {"Linear Regression", "SVR", "Fine Tree"}
PY
```

Expected: semua assertion lulus.

- [ ] **Step 5: Final commit**

```bash
git add requirements.txt README.md main.ipynb
git commit -m "chore: pin environment and verify reproducibility"
```

---

## Execution Order and Time Budget

| Task | Estimasi | Critical path |
|---|---:|---|
| 1. Preprocessing + tests | 30–45 menit | Tidak |
| 2. Download 2019–2023 | 30–45 menit | **Ya** |
| 3. Notebook EDA | 30 menit | Tidak |
| 4. Sanity check | 30 menit | Tidak |
| 5. t+1 dataset + LR | 30 menit | Ya |
| 6. SVR/Tree/RF tuning | 30–60 menit | Ya |
| 7. Importance + error | 30–45 menit | Tidak |
| 8. Report | 45–60 menit | Ya |
| 9. Reproducibility | 20 menit | Ya |

Jika waktu tersisa setelah Task 7, tambahkan hanya satu hal: plot partial dependence untuk `wind_speed_lag0` dan `temperature_50m`. Jangan menambah model baru.

## Handoff Checklist

- [ ] `AGENTS.md` dan design doc tidak diubah.
- [ ] `data/raw/` tidak masuk git.
- [ ] `results/` berisi semua CSV dan figure yang dikutip laporan.
- [ ] Notebook dieksekusi dari atas ke bawah tanpa error.
- [ ] Tidak ada angka di `report.md` yang tidak berasal dari `results/`.
- [ ] Keterbatasan dan perbedaan dengan paper dijelaskan jujur.
