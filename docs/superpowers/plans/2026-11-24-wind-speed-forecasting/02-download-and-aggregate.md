# Plan 02 — Download and Aggregate 2019–2023

> **Plan 02 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 01 lulus (6 tests).

**Goal:** Mengunduh data MIDC NWTC M2 per bulan 2019–2023, mem-parse dengan modul Plan 01, dan menghasilkan `daily.csv`, `monthly.csv`, serta `data_quality.csv`.

**Depends on:** Plan 01 (`parse_midc`, `aggregate_daily`, `aggregate_monthly`).

**Files:**
- Create: `scripts/download_data.py`
- Generated: `data/processed/daily.csv`, `data/processed/monthly.csv`, `data/processed/data_quality.csv`
- Generated (gitignored): `data/raw/nwtc_YYYY_MM.csv`

**Interfaces:**
- Consumes: `parse_midc`, `aggregate_daily`, `aggregate_monthly`.
- Produces: `fetch_month(year: int, month: int, raw_dir: Path, retries: int = 3) -> Path`, `main(start=(2019, 1), end=(2023, 12)) -> None`.

**Constraints khusus task ini:**
- Download per bulan, bukan setahun penuh. API ±27 detik/bulan dan bisa timeout untuk rentang panjang.
- Resume dari file yang sudah lengkap; tulis atomic (`.part` lalu `replace`).
- Setelah concat, `daily` di-reindex ke 1.826 tanggal dan `monthly` ke 60 periode.
- Jangan commit `data/raw/`.

---

- [ ] **Step 1: Write `scripts/download_data.py`**

Wajib memiliki:

- `BASE_URL = "https://midcdmz.nlr.gov/apps/data_api.pl"`.
- `fetch_month`: hitung `start` = hari pertama bulan, `end` = akhir bulan; request dengan `params={"site": "NWTC", "begin": ..., "end": ...}`, `timeout=180`.
- Resume: jika file sudah ada dan jumlah baris data `>= 99%` dari `days_in_month * 1440`, lewati download.
- Retry 3× dengan backoff 5/10 detik.
- Tulis ke `.part` lalu `replace()` agar atomic.
- Validasi respons: status 2xx dan jumlah baris `>= 99%` dari ekspektasi. Jangan menerima file terpotong.
- `main`: loop 60 bulan, parse tiap bulan, catat quality metrics, agregasi, gabungkan.
- Setelah concat:
  ```python
  daily = daily.reindex(pd.date_range("2019-01-01", "2023-12-31", freq="D"))
  daily.index.name = "date"
  monthly = monthly.reindex(pd.period_range("2019-01", "2023-12", freq="M"))
  monthly.index.name = "month"
  ```
- Simpan CSV dengan `float_format="%.6f"`.
- `data_quality.csv` minimal berisi: `year, month, expected_minutes, n_rows, n_missing_minutes, valid_wind_speed, valid_wind_dir, valid_temperature, valid_humidity, valid_pressure, valid_precipitation`.
- Import dari `scripts.preprocess`; jalankan sebagai module (`python -m`), bukan `python scripts/download_data.py`.

- [ ] **Step 2: Smoke test one month**

File `data/raw/nwtc_2023_01.csv` kemungkinan sudah ada dari verifikasi awal. `fetch_month` harus melewati download dan langsung mem-parse.

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

Expected: 60 bulan diproses; ±27 menit. Jika dijalankan di background, tunggu sampai selesai sebelum lanjut. Jika satu bulan gagal, rerun akan resume dari file yang sudah lengkap.

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

**Verification gate:** `daily.csv` 1.826 baris, `monthly.csv` 60 baris, `data_quality.csv` 60 baris, tanpa tanggal duplikat. Jangan lanjut ke Plan 03 sebelum ini lulus.
