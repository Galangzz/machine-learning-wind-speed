# Daily t+1 Wind Speed Forecasting — Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each task plan **in order**. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun notebook Jupyter yang reproducible dan laporan Markdown berbahasa Indonesia yang mereplikasi Altork (2025) sebagai sanity check, lalu mengevaluasi peramalan kecepatan angin harian satu hari ke depan (t+1) di NWTC M2.

**Architecture:** Downloader MIDC yang checkpointed mengubah export CSV 1 menit per bulan menjadi agregat harian dan bulanan. Notebook memuat agregat tersebut, menjalankan sanity check bulanan, membangun dataset supervised t+1, melakukan tuning dan evaluasi lima predictor, lalu menyimpan semua metrik/gambar ke `results/`. Laporan hanya mengutip angka dari `results/`.

**Tech Stack:** Python 3.14 (uv), pandas 3.1, numpy 2.5, scikit-learn 1.9, matplotlib, seaborn, Jupyter.

**Spec:** `docs/superpowers/specs/2026-11-24-wind-speed-forecasting-design.md`

**Operating rules:** `AGENTS.md`

**Target internal:** 28 November 2026. **Deadline:** 1 Desember 2026. Hari 29 Nov–1 Des menjadi buffer.

---

## Urutan Task

Kerjakan **berurutan 01 → 09**. Jangan mulai task berikutnya sebelum verifikasi task sebelumnya lulus dan commit-nya ada.

| # | Plan | Estimasi | Critical path | Depends on |
|---|---|---:|---|---|
| 01 | [Preprocessing core + tests](01-preprocessing.md) | 30–45 menit | Tidak | — |
| 02 | [Download & aggregate 2019–2023](02-download-and-aggregate.md) | 30–45 menit | **Ya** | 01 |
| 03 | [Notebook EDA & data quality](03-notebook-eda.md) | 30 menit | Tidak | 02 |
| 04 | [Monthly sanity check](04-sanity-check-replication.md) | 30 menit | Tidak | 03 |
| 05 | [t+1 dataset + persistence/LR](05-t1-dataset-and-linear-baseline.md) | 30 menit | Ya | 04 |
| 06 | [SVR, Fine Tree, Random Forest](06-nonlinear-models.md) | 30–60 menit | Ya | 05 |
| 07 | [Feature importance & error analysis](07-feature-importance-and-error-analysis.md) | 30–45 menit | Tidak | 06 |
| 08 | [Indonesian report](08-indonesian-report.md) | 45–60 menit | Ya | 07 |
| 09 | [Reproducibility & final verification](09-reproducibility.md) | 20 menit | Ya | 08 |

Critical path total: download (±27 menit) + modeling + laporan. Target selesai 28 November; buffer 29 November–1 Desember.

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
- Arah angin: vector mean `atan2(mean(sin), mean(cos))`, lalu encode sin/cos. Jangan rata-ratakan derajat secara linear. 360° = 0°.
- Agregat bulanan dihitung langsung dari 1 menit, bukan dari rata-rata harian.
- Split eksperimen utama berdasarkan `target_date`: train ≤ 2021-12-31, validation 2022, test 2023. Tidak ada shuffling. **Pengecualian:** sanity check replikasi memakai random split 70/15/15 dengan `random_state=42`, sesuai paper.
- Scaler fit pada train saja; saat refit final, fit pada train+validation saja.
- `random_state=42` di semua split, model, dan permutation importance.
- Model: persistence, Linear Regression, SVR RBF, Fine Tree (`DecisionTreeRegressor`), Random Forest. Tidak ada LSTM/GRU/Transformer/XGBoost/stacking.
- Metrik: RMSE, MSE, MAE, R². RMSE dan MAE reduction terhadap persistence wajib dihitung.
- Laporan `report.md` berbahasa Indonesia. Semua angka berasal dari `results/*.csv`.
- Jangan commit `data/raw/`, `.venv/`, `.ipynb_checkpoints/`.
- Notebook harus dieksekusi dari atas ke bawah tanpa hidden state.

## Review Focus

Lima kegagalan yang paling mungkin terjadi dan harus ditutup oleh test/audit:

1. **`MST` dibaca sebagai menit** → timestamp duplikat, tanggal bergeser, dan file Januari 2023 tidak lagi 44.640 baris. Test: `test_parse_midc_treats_mst_as_hhmm` (Plan 01).
2. **Presipitasi dijumlahkan atau nilai 00:00 ikut dihitung** → total harian ribuan mm, atau hari kering setelah hujan mendapat nilai hujan hari sebelumnya. Test: `test_daily_precipitation_ignores_midnight_stale_value` (Plan 01).
3. **Arah angin dirata-ratakan linear** → 350° dan 10° menjadi 180°, bukan 0°. Test: `test_daily_wind_direction_uses_vector_mean` (Plan 01).
4. **Hari dengan data sedikit tetap dianggap lengkap** → satu jam data mendapat bobot sama dengan satu hari penuh. Test: `test_daily_completeness_threshold_is_75_percent` (Plan 01).
5. **Leakage** → random split, scaler fit pada seluruh data, atau meteorologi hari t+1 masuk fitur. Audit eksplisit di Plan 05.

## Verified Data Facts

Sudah diverifikasi langsung pada 24 November 2026:

- Januari 2023: 44.640 baris data, tepat 31 hari × 1.440 menit, tanpa duplikat setelah `MST` di-parse sebagai HHMM. Rentang: 2023-01-01 00:00–2023-01-31 23:59.
- Februari 2020 (tahun kabisat): 41.760 baris, tepat 29 hari × 1.440 menit. Rentang: 2020-02-01 00:00–2020-02-29 23:59.
- Enam kolom modeling tidak memiliki sentinel atau NA pada kedua sampel tersebut, tetapi kolom lain memilikinya; parser tetap wajib menangani sentinel.
- Presipitasi adalah akumulasi harian yang reset pada 00:01. Contoh stale value:
  - 2023-01-03 00:00 = 2,286 mm (sisa 2 Januari); hujan aktual 3 Januari = 0.
  - 2023-01-19 00:00 = 6,096 mm (sisa 18 Januari); hujan aktual 19 Januari = 0.
  - 2020-02-04 00:00 = 5,588 mm (sisa 3 Februari); hujan aktual 4 Februari = 1,778 mm.
- API: satu hari ±8 detik; satu bulan ±27 detik dan 18–22 MB. Full 60 bulan ±27 menit. Jangan query setahun penuh sekaligus (pernah timeout dan menghasilkan file terpotong).

## Cara Memakai Plan Ini

- Baca `AGENTS.md` dan file ini sebelum mulai.
- Setiap plan task memuat langkah, verifikasi, dan commit sendiri; constraint global tetap berlaku.
- Jalankan verifikasi di akhir setiap plan. Kalau gagal, perbaiki dulu; jangan lanjut.
- Setelah setiap task sukses, commit dengan pesan yang ditentukan di plan task tersebut.
- Jangan mengubah design doc, `AGENTS.md`, atau plan ini tanpa diminta.

## Handoff Checklist

- [ ] `AGENTS.md` dan design doc tidak diubah.
- [ ] `data/raw/` tidak masuk git.
- [ ] `results/` berisi semua CSV dan figure yang dikutip laporan.
- [ ] Notebook dieksekusi dari atas ke bawah tanpa error.
- [ ] Tidak ada angka di `report.md` yang tidak berasal dari `results/`.
- [ ] Keterbatasan dan perbedaan dengan paper dijelaskan jujur.
