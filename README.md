# Peramalan Kecepatan Angin Harian t+1 — NWTC M2

Replikasi ringan Altork (2025) sebagai *sanity check*, lalu ekstensi utama: peramalan
kecepatan angin harian satu hari ke depan (t+1) di NREL/NLR National Wind Technology
Center M2, Boulder, Colorado (2019–2023).

## Isi repo

| Path | Keterangan |
|---|---|
| `scripts/download_data.py` | Pengunduh MIDC per bulan (checkpoint/resume) + agregasi harian/bulanan |
| `scripts/preprocess.py` | Parsing dan agregasi murni (MST HHMM, sentinel, presipitasi, arah angin sirkular, completeness) |
| `tests/` | Regression test untuk jebakan data dan perilaku pengunduh |
| `main.ipynb` | Notebook deliverable: EDA, sanity check, dataset t+1, modeling, analisis |
| `report.md` | Laporan analisis (bahasa Indonesia) |
| `data/processed/` | `daily.csv`, `monthly.csv`, `data_quality.csv` |
| `results/` | CSV metrik, `best_params.json`, dan `results/figures/` |

## Menjalankan dari nol

```bash
uv sync
uv run python -m scripts.download_data   # ±36 menit untuk 60 bulan; resume otomatis
uv run pytest -q
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

`data/raw/` tidak di-commit (sekitar 20 MB per bulan). Notebook hanya membutuhkan
`data/processed/*.csv`, sehingga bisa dieksekusi ulang tanpa mengunduh.

## Konfigurasi eksperimen

- Split kronologis berdasarkan `target_date`: latih 2019–2021 (1.078 baris), validasi 2022 (365), uji 2023 (365).
- Fitur (16): `wind_speed_lag0..7`, `wind_dir_sin/cos`, `temperature_50m`, `pressure`, `relative_humidity`, `precipitation`, `doy_sin/cos`.
- Scaler Min-Max [−1, 1] hanya di-fit pada data latih; model final di-fit pada latih+validasi.
- `random_state = 42` di semua split, model, pencarian hyperparameter, dan permutation importance.
- Sanity check replikasi memakai split acak 70/15/15 (meniru paper), bukan split kronologis.
