# AGENTS.md

## Project

Tugas Akhir: replikasi ringan Altork (2025) sebagai *sanity check*, lalu ekstensi utama berupa **daily one-day-ahead (t+1) wind speed forecasting** menggunakan data NREL/NLR NWTC M2, Boulder, Colorado.

- Paper: Altork, Y. (2025). *Comparative analysis of machine learning models for wind speed forecasting: Support vector machines, fine tree, and linear regression approaches.* International Journal of Thermofluids, 27, 101217. https://doi.org/10.1016/j.ijft.2025.101217
- Dataset: Jager, D.; Andreas, A. (1996). *NREL National Wind Technology Center (NWTC): M2 Tower; Boulder, Colorado (Data)*. NREL Report No. DA-5500-56489. https://www.osti.gov/biblio/1052222
- Sumber data: https://midcdmz.nlr.gov/nwtc_m2/ (domain lama `midcdmz.nrel.gov` redirect ke `midcdmz.nlr.gov`)
- API: `https://midcdmz.nlr.gov/apps/data_api.pl?site=NWTC&begin=YYYYMMDD&end=YYYYMMDD`

## Source of truth

1. **Design/scope:** `docs/superpowers/specs/2026-11-24-wind-speed-forecasting-design.md`
2. **Execution plan:** `docs/superpowers/plans/2026-11-24-wind-speed-forecasting.md`
3. **Operating rules:** file ini.

Jika terjadi konflik: design doc menang untuk scope, plan menang untuk urutan eksekusi, file ini menang untuk aturan operasional.

## Environment

- Python 3.14 via `uv` (`.python-version`).
- Dependency di `pyproject.toml`; `uv.lock` **wajib** di-commit.
- Perintah:
  - `uv sync`
  - `uv run pytest -q`
  - `uv run python -m scripts.download_data`
  - `uv run jupyter lab`
  - Cek notebook headless: `uv run jupyter nbconvert --to notebook --execute main.ipynb --output /tmp/opencode/main_executed.ipynb`
- CPU only. Tidak ada GPU, tidak ada deep learning.

## Aturan data yang tidak boleh dilanggar

- **`MST` adalah HHMM, bukan menit sejak tengah malam.** `hour = MST // 100`, `minute = MST % 100`. Contoh: `MST=100` berarti 01:00, bukan 01:40. Kesalahan ini membuat timestamp duplikat dan rentang tanggal bergeser.
- Kolom modeling:
  - `Avg Wind Speed @ 50m [m/s]` → `wind_speed_50m`
  - `Avg Wind Direction @ 50m [deg]` → `wind_dir_50m`
  - `Temperature @ 50m [deg C]` → `temperature_50m`
  - `Relative Humidity [%]` → `relative_humidity`
  - `Station Pressure [mBar]` → `pressure`
  - `Precipitation (Accumulated) [mm]` → `precipitation`
- Sentinel `-9999`, `-7999`, `-6999`, `-999` → `NaN`.
- **Completeness harian:** minimal 75% menit valid (1.080 dari 1.440 menit). Di bawah itu, agregat harian variabel tersebut adalah `NaN`.
- **Presipitasi adalah akumulasi harian yang reset sekitar 00:01.** Nilai pada 00:00 adalah sisa hari sebelumnya (*stale*). **Buang menit 00:00, lalu ambil `max` harian.** Jangan pernah menjumlahkan kolom ini; `sum` akan menghasilkan ribuan mm.
- **Arah angin adalah data sirkular.** Gunakan *vector mean*: `atan2(mean(sin(dir)), mean(cos(dir)))`, lalu encode sin/cos. Jangan rata-ratakan derajat secara linear. 360° = 0°.
- **Agregat bulanan dihitung langsung dari data 1 menit**, bukan dari rata-rata harian.
- Jangan menginterpolasi `wind_speed_50m` atau target. Drop baris dengan target atau lag yang hilang.
- **Kontrak fitur/target:** fitur hanya memakai data sampai akhir hari t (23:59); target adalah `wind_speed_50m` hari t+1. Meteorologi hari t+1 tidak boleh masuk fitur.

## Aturan modeling yang tidak boleh dilanggar

- Split kronologis berdasarkan **tanggal target**: train 2019-01-01–2021-12-31, validation 2022, test 2023. Tidak ada shuffling.
- Scaler di-fit pada train saja. Saat refit pada train+validation, fit scaler baru pada train+validation saja.
- Baseline **persistence** (`pred(t+1) = wind_speed(t)`) wajib ada.
- `random_state=42` di semua tempat.
- Metrik: RMSE, MSE, MAE, R². Selalu bandingkan dengan persistence.
- Scope model: persistence, Linear Regression, SVR (RBF), Fine Tree (`DecisionTreeRegressor`), Random Forest. Tidak ada LSTM/GRU/Transformer. XGBoost/stacking di luar scope untuk deadline ini.
- Setiap angka di `report.md` harus berasal dari `results/*.csv` atau output notebook. Tidak ada angka karangan atau estimasi manual.

## Struktur repo

- `data/raw/` — export CSV MIDC per bulan; **gitignored**; sekitar 20 MB/bulan.
- `data/processed/daily.csv` — agregat harian, satu baris per tanggal.
- `data/processed/monthly.csv` — agregat bulanan, 60 baris.
- `data/processed/data_quality.csv` — laporan completeness/sentinel per bulan.
- `results/` — CSV metrik, `best_params.json`, dan `results/figures/`.
- `scripts/download_data.py` — CLI downloader dengan checkpoint/resume.
- `scripts/preprocess.py` — fungsi parsing/agregasi murni.
- `tests/test_preprocess.py` — regression test untuk jebakan data di atas.
- `main.ipynb` — notebook deliverable.
- `report.md` — laporan Markdown berbahasa Indonesia.

## Working agreement

- Kerjakan plan task-by-task; commit setelah setiap task dengan pesan yang deskriptif.
- Jangan mengubah design doc atau file ini tanpa diminta.
- Jangan commit `data/raw/`, `.venv/`, atau notebook checkpoint.
- Kalau hasil bertentangan dengan ekspektasi, laporkan apa adanya; jangan tuning agar cocok dengan paper.
- API lambat (±25–30 detik per bulan). Download per bulan, simpan checkpoint, dan resume dari file yang sudah ada.
- Notebook harus bisa dieksekusi dari atas ke bawah tanpa hidden state.
- Target internal: **28 November 2026**. Deadline: **1 Desember 2026**; 29 November–1 Desember adalah buffer.

## Definition of done

- `uv run pytest -q` lulus.
- `main.ipynb` dieksekusi headless tanpa error.
- `daily.csv` mencakup 2019-01-01–2023-12-31 (1.826 baris sebelum drop; laporkan jumlah aktual).
- `monthly.csv` memiliki 60 baris.
- Tabel sanity check (3 model) dan tabel utama (persistence + 4 model) ada di `results/`.
- Analisis feature importance dan error per musim/level wind speed ada.
- `report.md` berbahasa Indonesia dan konsisten dengan `results/`.
