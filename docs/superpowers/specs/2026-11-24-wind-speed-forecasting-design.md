# Rancangan Tugas Akhir — Daily One-Day-Ahead Wind Speed Forecasting

**Status:** Draft 1 untuk direview
**Tanggal:** 24 November 2026
**Deadline:** 1 Desember 2026
**Target internal:** 30 November 2026
**Paper acuan:** Altork, Y. (2025). *Comparative analysis of machine learning models for wind speed forecasting: Support vector machines, fine tree, and linear regression approaches.* International Journal of Thermofluids, 27, 101217. https://doi.org/10.1016/j.ijft.2025.101217

---

## 1. Ringkasan

Tugas akhir ini melakukan **replikasi ringan** paper Altork (2025) sebagai sanity check, lalu **ekstensi utama** berupa peramalan kecepatan angin harian satu hari ke depan (*daily one-day-ahead forecasting*, t+1).

Paper asli membandingkan SVM/SVR, Fine Tree, dan Linear Regression pada data meteorologi **bulanan** NWTC M2 (2019–2023, 60 observasi). Berdasarkan metodologi yang dilaporkan, paper tersebut melakukan regresi pada periode yang sama (*same-period regression*): lima variabel meteorologi dipetakan ke wind speed pada periode yang sama, tanpa fitur historis. Angka R² = 0,97 yang dilaporkan sebagian besar mencerminkan smoothing bulanan dan sifat same-period tersebut, bukan kemampuan peramalan operasional.

Ekstensi yang diusulkan:

1. Mengagregasi data 1 menit NWTC M2 menjadi data harian 2019–2023 (sekitar 1.826 observasi).
2. Mendefinisikan tugas peramalan yang operasional: pada akhir hari t, prediksi rata-rata wind speed 50 m untuk hari t+1.
3. Membandingkan tiga model paper dengan **Random Forest** sebagai model ensemble non-linear tambahan, plus **persistence baseline**.
4. Menganalisis performa, feature importance, dan pola error berdasarkan musim serta level kecepatan angin.

Hasil yang diharapkan bukan meniru angka paper, melainkan menjawab pertanyaan: **apakah model yang menang pada data bulanan same-period juga menang pada peramalan harian t+1, dan apakah model ensemble non-linear memberikan nilai tambah?**

---

## 2. Latar Belakang dan Masalah

### 2.1 Paper acuan

- Tiga model: SVM/SVR, Fine Tree, Linear Regression.
- Lima input: atmospheric pressure, ambient temperature, wind direction, relative humidity, precipitation.
- Target: wind speed pada ketinggian 50 m.
- Data: NWTC M2, Boulder, Colorado, rata-rata bulanan 2019–2023, 60 observasi.
- Split: 70% training, 15% validation, 15% testing.
- Normalisasi: Min-Max ke rentang [-1, 1].
- Metrik: RMSE, MSE, MAE, R².
- Hasil: Linear Regression terbaik (RMSE 0,29555; R² 0,97), lalu SVM (RMSE 0,32275; R² 0,96), lalu Fine Tree (RMSE 0,44042; R² 0,93).

### 2.2 Keterbatasan yang menjadi dasar ekstensi

Paper sendiri mengakui:

- Rata-rata bulanan menghilangkan variasi jangka pendek.
- Data harian atau per jam disarankan untuk penelitian selanjutnya.
- Feature engineering diperlukan untuk interval waktu yang lebih pendek.
- Ensemble learning dan composite models disebut sebagai arah pengembangan.

### 2.3 Pertanyaan riset

1. Apakah model yang menang pada data bulanan (same-period) juga menang pada peramalan harian t+1?
2. Apakah Random Forest sebagai model ensemble non-linear mengungguli Linear Regression, SVR, dan Fine Tree pada tugas t+1?
3. Fitur apa yang paling berpengaruh pada tugas t+1, dan apakah temperatur masih dominan seperti klaim paper?
4. Seberapa besar penurunan performa ketika tugas diubah dari same-period regression menjadi peramalan t+1?

---

## 3. Tujuan dan Non-Tujuan

### 3.1 Tujuan

1. Mengimplementasikan pipeline paper (tiga model, lima fitur, empat metrik) pada data bulanan sebagai sanity check.
2. Membangun dataset harian 2019–2023 dari data 1 menit NWTC M2.
3. Mendefinisikan dan mengevaluasi tugas peramalan t+1 dengan split kronologis.
4. Membandingkan persistence baseline, Linear Regression, SVR, Fine Tree, dan Random Forest.
5. Menganalisis feature importance dan error berdasarkan musim serta level wind speed.
6. Menyusun notebook Jupyter end-to-end dan laporan analisis Markdown.

### 3.2 Non-tujuan

- Tidak mengejar reproduksi angka paper secara identik.
- Tidak membangun sistem produksi atau API deployment.
- Tidak menggunakan LSTM/GRU/Transformer.
- Tidak menggunakan data prakiraan cuaca (NWP) untuk hari target.
- Tidak melakukan generalisasi multi-lokasi.
- Tidak menggunakan resolusi per jam.
- Tidak melakukan optimasi GA atau hyperparameter search yang ekstensif.

### 3.3 Anti-goals

- Tidak boleh ada data leakage: tidak boleh random split untuk data time series, tidak boleh fit scaler pada seluruh data, tidak boleh memakai meteorologi hari t+1.
- Tidak boleh mengklaim paper salah; posisikan sebagai ekstensi dan koreksi protokol.
- Tidak boleh memfabrikasi hasil atau angka.
- Tidak boleh menambah model hanya demi jumlah model tanpa justifikasi referensi.

---

## 4. Data

### 4.1 Sumber

- Dataset: NREL National Wind Technology Center (NWTC) M2 Tower, Boulder, Colorado.
- Pengelola saat ini: NLR Flatirons Campus (M2), Measurement and Instrumentation Data Center (MIDC).
- Halaman data: `https://midcdmz.nlr.gov/nwtc_m2/`
- API publik: `https://midcdmz.nlr.gov/apps/data_api.pl?site=NWTC&begin=YYYYMMDD&end=YYYYMMDD`
- Catatan: domain lama `midcdmz.nrel.gov` sekarang redirect ke `midcdmz.nlr.gov`.
- Sitasi: Jager, D.; Andreas, A. (1996). *NREL National Wind Technology Center (NWTC): M2 Tower; Boulder, Colorado (Data)*. NREL Report No. DA-5500-56489. https://www.osti.gov/biblio/1052222

### 4.2 Periode dan resolusi

- Periode: 1 Januari 2019 – 31 Desember 2023.
- Resolusi asli: rata-rata 1 menit.
- Resolusi analisis: harian.
- Jumlah hari teoretis: 1.826 hari (2020 tahun kabisat).
- Jumlah observasi bulanan untuk sanity check: 60 bulan.

### 4.3 Variabel

| Peran | Variabel | Kolom MIDC | Agregasi harian |
|---|---|---|---|
| Target | Wind speed @ 50 m | `Avg Wind Speed @ 50m [m/s]` | mean |
| Fitur | Wind speed lag 1–7 | kolom yang sama | lag dari daily mean |
| Fitur | Wind direction @ 50 m | `Avg Wind Direction @ 50m [deg]` | vector mean, lalu sin/cos |
| Fitur | Temperature @ 50 m | `Temperature @ 50m [deg C]` | mean |
| Fitur | Relative humidity | `Relative Humidity [%]` | mean |
| Fitur | Station pressure | `Station Pressure [mBar]` | mean |
| Fitur | Precipitation | `Precipitation (Accumulated) [mm]` | total harian (verifikasi dari data mentah) |
| Fitur | Kalender | tanggal | day-of-year sin/cos |

### 4.4 Aturan agregasi

- **Completeness threshold harian:** minimal 75% menit valid (1.080 dari 1.440 menit) untuk menghasilkan agregat harian. Jika kurang, agregat harian di-set NaN. Ini mencegah daily mean dari beberapa menit data saja.
- Variabel kontinu (wind speed, temperature, RH, pressure): rata-rata semua menit valid dalam satu hari.
- Wind direction: hitung rata-rata vektor:
  - `x = mean(sin(dir))`, `y = mean(cos(dir))`
  - `dir_mean = atan2(y, x)` dikonversi ke [0, 360)
  - Simpan `wind_dir_sin = sin(dir_mean)`, `wind_dir_cos = cos(dir_mean)`.
- Precipitation: tentukan apakah kolom akumulatif reset per hari. Jika ya, ambil nilai akhir hari atau selisih; jika tidak, jumlahkan. Metode final harus diverifikasi pada beberapa hari hujan dan didokumentasikan.
- **Agregasi bulanan (sanity check):** hitung rata-rata bulanan langsung dari data 1 menit, bukan dari rata-rata harian. Untuk wind direction, gunakan vector mean. Ini lebih dekat ke cara paper mengagregasi data bulanan. Laporkan juga jumlah menit valid per bulan.
- Hari dengan target wind speed hilang: drop dari dataset modeling.
- Hari dengan fitur hilang: interpolasi linear dengan batas gap pendek (misalnya maksimal 2 hari), lalu drop sisa NaN. Jumlah hari yang di-drop harus dilaporkan.

### 4.5 Data quality checks

- Jumlah hari per tahun dan jumlah hari hilang.
- Rentang nilai tiap variabel; deteksi sentinel (-9999, -7999, -6999, dan nilai tidak wajar).
- Duplikasi timestamp.
- Distribusi target: min, max, mean, median, standar deviasi.
- Plot time series target dan fitur utama.
- Perbandingan statistik deskriptif dataset bulanan hasil agregasi dengan Tabel 1 paper (sebagai sanity check, bukan syarat identik).

### 4.6 File data

- `data/raw/` — file mentah per bulan (opsional, boleh dihapus setelah agregasi).
- `data/processed/daily.csv` — dataset harian untuk eksperimen utama.
- `data/processed/monthly.csv` — dataset bulanan untuk sanity check.
- `data/processed/data_quality.csv` — ringkasan missing data dan statistik dasar.

---

## 5. Definisi Tugas Peramalan

### 5.1 Waktu penerbitan forecast

Forecast dianggap diterbitkan pada **akhir hari t (23:59 MST)**, untuk memprediksi rata-rata wind speed hari t+1.

Konsekuensi:

- Semua observasi sampai akhir hari t boleh dipakai.
- Meteorologi hari t+1 **tidak boleh** dipakai.
- Fitur meteorologi yang dipakai adalah nilai hari t (dan lag sebelumnya).
- Target adalah `wind_speed_50m(t+1)`.

### 5.2 Feature set

Fitur final (16 fitur):

1. `wind_speed_lag0` = wind speed hari t
2. `wind_speed_lag1` = wind speed hari t-1
3. `wind_speed_lag2` = wind speed hari t-2
4. `wind_speed_lag3` = wind speed hari t-3
5. `wind_speed_lag4` = wind speed hari t-4
6. `wind_speed_lag5` = wind speed hari t-5
7. `wind_speed_lag6` = wind speed hari t-6
8. `wind_speed_lag7` = wind speed hari t-7
9. `wind_dir_sin` = sin(arah angin hari t)
10. `wind_dir_cos` = cos(arah angin hari t)
11. `temperature` = temperature hari t
12. `pressure` = pressure hari t
13. `relative_humidity` = RH hari t
14. `precipitation` = presipitasi hari t
15. `doy_sin` = sin(2π × day-of-year(t+1) / 365.25)
16. `doy_cos` = cos(2π × day-of-year(t+1) / 365.25)

Catatan: fitur kalender mengacu pada hari target (t+1), karena yang dimodelkan adalah musim dari nilai yang diprediksi. Tanggal t+1 sudah diketahui pada saat forecast diterbitkan.

Opsional jika waktu tersisa: rolling mean 3 dan 7 hari dari wind speed. Jangan dimasukkan ke core kalau belum diuji.

### 5.3 Pembentukan dataset supervised

- Urutkan data berdasarkan tanggal.
- Buat `target = wind_speed.shift(-1)` (nilai hari t+1).
- Buat `wind_speed_lag0..7 = wind_speed.shift(0..7)`.
- Fitur meteorologi dan wind direction memakai nilai hari t (tidak di-shift).
- Drop baris dengan target atau lag yang hilang (7 hari pertama dan 1 hari terakhir).
- Pastikan tidak ada nilai hari t+1 yang masuk ke fitur.
- Laporkan jumlah baris final setelah drop.

### 5.4 Split kronologis

| Split | Periode | Perkiraan jumlah hari | Fungsi |
|---|---|---|---|
| Train | 2019-01-01 – 2021-12-31 | 1.096 | fitting model dan hyperparameter tuning |
| Validation | 2022-01-01 – 2022-12-31 | 365 | pemilihan model/hyperparameter |
| Test | 2023-01-01 – 2023-12-31 | 365 | evaluasi final, tidak boleh dipakai untuk tuning |

Aturan:

- Tidak ada shuffling.
- Scaler di-fit hanya pada train.
- Hyperparameter dipilih berdasarkan validation (atau TimeSeriesSplit di dalam train).
- Test hanya disentuh sekali untuk evaluasi final.
- Lag boleh melewati batas split karena lag hanya melihat ke belakang; ini sah dalam peramalan. Namun, pastikan tidak ada fitur yang melihat ke depan.

---

## 6. Model

### 6.1 Daftar model

| Model | Implementasi | Scaling | Hyperparameter utama |
|---|---|---|---|
| Persistence | `pred(t+1) = y(t)` | tidak | tidak ada |
| Linear Regression | `LinearRegression` | ya, Min-Max [-1, 1] | tidak ada |
| SVR | `SVR(kernel="rbf")` | ya, Min-Max [-1, 1] | C, gamma, epsilon |
| Fine Tree | `DecisionTreeRegressor` | tidak | max_depth, min_samples_leaf |
| Random Forest | `RandomForestRegressor` | tidak | n_estimators, max_depth, min_samples_leaf, max_features |

### 6.2 Justifikasi penambahan Random Forest

- Paper Altork (2025) menyebut Random Forest sebagai salah satu metode yang terbukti memberikan akurasi tinggi untuk prediksi kondisi cuaca (bagian pendahuluan).
- Paper menyebut ensemble learning dan composite models sebagai future work.
- Random Forest adalah model ensemble non-linear yang kuat untuk data tabular, tahan terhadap outlier, dan memberikan feature importance.
- Referensi algoritma: Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

### 6.3 Hyperparameter tuning

- Gunakan `TimeSeriesSplit` (misalnya 3 fold) di dalam training set untuk model yang punya hyperparameter.
- Grid kecil, bukan pencarian besar:
  - SVR: C ∈ {0.1, 1, 10, 100}; gamma ∈ {"scale", 0.01, 0.1, 1}; epsilon ∈ {0.01, 0.1, 0.5}.
  - Fine Tree: max_depth ∈ {None, 5, 10, 20}; min_samples_leaf ∈ {1, 2, 4, 8}.
  - Random Forest: n_estimators ∈ {200, 500}; max_depth ∈ {None, 10, 20}; min_samples_leaf ∈ {1, 2, 4}; max_features ∈ {"sqrt", 1.0}.
- Evaluasi konfigurasi terbaik pada validation set.
- Refit model terbaik pada train + validation, lalu evaluasi di test.
- Catat konfigurasi final di laporan.

### 6.4 Baseline persistence

Persistence adalah baseline wajib:

- Prediksi hari t+1 = wind speed hari t.
- Jika model ML tidak mengalahkan persistence, itu temuan penting dan harus dilaporkan.
- Hitung RMSE/MAE reduction terhadap persistence.

---

## 7. Evaluasi

### 7.1 Metrik

- RMSE
- MSE
- MAE
- R²

Semua dihitung pada test set (2023) dan, untuk analisis tambahan, pada validation set.

### 7.2 Perbandingan dengan persistence

- RMSE reduction (%) = `(RMSE_persistence - RMSE_model) / RMSE_persistence × 100`
- MAE reduction (%) = `(MAE_persistence - MAE_model) / MAE_persistence × 100`
- Nilai positif berarti model lebih baik dari persistence.

### 7.3 Analisis tambahan

- **Feature importance:**
  - Random Forest: `feature_importances_`.
  - Permutation importance pada validation/test untuk model terbaik.
  - Linear Regression: koefisien (setelah scaling, interpretasi hati-hati).
- **Error analysis:**
  - RMSE/MAE per musim: DJF, MAM, JJA, SON.
  - RMSE/MAE per level wind speed: rendah, sedang, tinggi (misalnya tertile target test).
  - Residual distribution.
- **Visualisasi:**
  - Time series actual vs predicted pada test (mungkin subset beberapa bulan agar terbaca).
  - Scatter plot predicted vs actual.
  - Bar chart feature importance.
  - Bar chart error per musim.

### 7.4 Perbandingan dengan paper

Bandingkan secara konseptual, bukan angka mentah:

- Paper: same-period, monthly, R² 0,97.
- Ekstensi: t+1, daily, R² kemungkinan jauh lebih rendah.
- Diskusikan mengapa perbedaan terjadi: smoothing, horizon, noise, split, fitur historis.

---

## 8. Sanity Check Replikasi Paper

### 8.1 Tujuan

Memastikan pipeline data dan modeling berjalan, serta melihat apakah arah hasil paper masuk akal. Bukan untuk mengejar angka identik.

### 8.2 Setup

- Dataset: `monthly.csv`, 60 observasi (2019–2023).
- Target: monthly mean wind speed 50 m.
- Fitur: pressure, temperature, wind direction **dalam derajat (satu fitur)**, RH, precipitation. Gunakan satu fitur arah angin agar tetap lima input seperti paper.
- Split: 70% train, 15% validation, 15% test, random split dengan `random_state=42`.
- Model: Linear Regression, SVR (RBF, default/light tuning), DecisionTreeRegressor (Fine Tree).
- Metrik: RMSE, MSE, MAE, R².
- Normalisasi: Min-Max [-1, 1], fit pada train.

### 8.3 Pelaporan

- Tabel metrik ketiga model.
- Perbandingan dengan Tabel 2 paper.
- Catatan perbedaan: paper tidak menjelaskan hyperparameter, split, dan detail agregasi; angka tidak harus identik.
- Jika hasil berbeda jauh, jelaskan kemungkinan penyebabnya (agregasi, split, implementasi Fine Tree, normalisasi).

---

## 9. Deliverable

### 9.1 Notebook

Satu notebook utama, misalnya `main.ipynb`, dengan struktur:

1. Setup dan import.
2. Konfigurasi (path, seed, periode, split).
3. Load data dan data quality report.
4. EDA: distribusi target, time series, korelasi, musim.
5. Sanity check replikasi bulanan.
6. Feature engineering harian dan pembuatan lag.
7. Split kronologis dan scaling.
8. Training dan tuning model.
9. Evaluasi test dan perbandingan persistence.
10. Feature importance dan error analysis.
11. Simpan hasil (tabel CSV/JSON, gambar).
12. Ringkasan hasil.

Notebook harus bisa dijalankan dari atas ke bawah setelah `daily.csv` dan `monthly.csv` tersedia. Download data boleh berada di script terpisah (`scripts/download_data.py`) agar notebook tidak timeout.

### 9.2 Laporan Markdown

File `report.md` (atau `laporan.md`), ditulis dalam **bahasa Indonesia** dengan istilah teknis Inggris bila diperlukan, dengan struktur:

1. Judul dan abstrak.
2. Pendahuluan: latar belakang, masalah, gap, tujuan.
3. Tinjauan pustaka: paper acuan, peramalan wind speed, model ensemble.
4. Data dan preprocessing: sumber, periode, agregasi, missing data, EDA.
5. Metodologi: definisi t+1, fitur, split, model, tuning, metrik.
6. Hasil sanity check replikasi.
7. Hasil eksperimen utama: metrik, persistence comparison, feature importance, error analysis.
8. Diskusi: kenapa R² turun, model mana yang menang, perbandingan dengan paper, implikasi.
9. Keterbatasan.
10. Kesimpulan.
11. Referensi.
12. Lampiran: konfigurasi model, versi library, seed.

### 9.3 Reproducibility

- `requirements.txt` berisi versi library yang dipakai.
- Seed tetap: `random_state=42` untuk split dan model.
- Simpan konfigurasi split dan hyperparameter final.
- Simpan hasil metrik ke CSV/JSON agar bisa dicek ulang.
- Jangan mengandalkan hasil manual di laporan; semua angka harus berasal dari notebook.

---

## 10. Justifikasi Modifikasi dan Referensi

| Modifikasi | Alasan | Referensi |
|---|---|---|
| Data harian | Paper bulanan menghilangkan variasi jangka pendek; paper menyarankan daily/hourly | Altork (2025), future work |
| Lag features | Peramalan t+1 membutuhkan informasi historis; paper tidak memakai lag | Altork (2025), future work; Hyndman & Athanasopoulos (2021) |
| Random Forest | Ensemble non-linear; paper menyebut RF dan ensemble learning | Altork (2025); Breiman (2001) |
| Chronological split | Mencegah leakage pada data time series | Hyndman & Athanasopoulos (2021) |
| Persistence baseline | Standar pembanding pada peramalan | Hyndman & Athanasopoulos (2021) |
| Wind direction sin/cos | Arah angin adalah variabel sirkular; rata-rata derajat linear tidak valid | Fisher (1993) |
| Min-Max fit on train only | Mencegah leakage; paper melakukan normalisasi sebelum split | Altork (2025); praktik standar ML |

Referensi lengkap:

- Altork, Y. (2025). Comparative analysis of machine learning models for wind speed forecasting: Support vector machines, fine tree, and linear regression approaches. *International Journal of Thermofluids, 27*, 101217. https://doi.org/10.1016/j.ijft.2025.101217
- Jager, D., & Andreas, A. (1996). *NREL National Wind Technology Center (NWTC): M2 Tower; Boulder, Colorado (Data)*. NREL Report No. DA-5500-56489. https://www.osti.gov/biblio/1052222
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Fisher, N. I. (1993). *Statistical Analysis of Circular Data*. Cambridge University Press.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD '16*. (hanya jika XGBoost dipakai)

---

## 11. Rencana 7 Hari

Tujuh hari kerja: 24–30 November 2026. Tanggal 1 Desember 2026 adalah deadline akhir, dipakai sebagai buffer jika ada slip.

| Hari | Fokus | Output |
|---|---|---|
| 1 (24 Nov) | Setup repo, script download, mulai download data, EDA awal | `daily.csv`, `monthly.csv`, data quality report |
| 2 (25 Nov) | Selesaikan preprocessing, sanity check replikasi bulanan | Tabel metrik sanity check, catatan perbedaan |
| 3 (26 Nov) | Feature engineering t+1, split kronologis, LR/SVR/Tree | Metrik validation untuk tiga model |
| 4 (27 Nov) | Random Forest, tuning, persistence comparison | Metrik test lengkap, model final |
| 5 (28 Nov) | Feature importance, error analysis, plots | Grafik dan tabel analisis |
| 6 (29 Nov) | Draft laporan Markdown | `report.md` versi 1 |
| 7 (30 Nov) | Buffer, finalisasi notebook, cek reproducibility | Notebook bersih, laporan final |

Catatan: tanggal 1 Desember adalah deadline; hari ke-7 dipakai untuk buffer dan finalisasi.

---

## 12. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Download 1 menit lambat/timeout | Data tidak siap | Download per bulan, agregasi langsung, mulai hari ini; simpan checkpoint per bulan |
| Missing data tinggi | Dataset harian berkurang | Laporkan missing rate; interpolasi gap pendek; drop target yang hilang; dokumentasikan |
| Presipitasi akumulatif ambiguous | Fitur precipitation salah | Verifikasi beberapa hari hujan; bandingkan beberapa metode agregasi; dokumentasikan pilihan |
| Hasil R² jauh lebih rendah | Terlihat “gagal” | Framing sebagai temuan: perbedaan same-period vs t+1; bandingkan dengan persistence, bukan hanya paper |
| RF tidak mengungguli LR | Nilai tambah model dipertanyakan | Laporkan apa adanya; diskusikan bahwa linear/persistence kuat pada data ini; feature importance tetap memberi insight |
| Scope creep (XGBoost, LSTM, SHAP) | Waktu habis | Core model hanya 5 + persistence; stretch hanya jika hari 5 selesai |
| Notebook tidak reproducible | Sulit dinilai | Seed tetap, `requirements.txt`, simpan konfigurasi, jalankan ulang dari atas ke bawah |

---

## 13. Keputusan yang Diserahkan ke Builder

### 13.1 Keputusan yang sudah diambil (my call)

Hal-hal berikut sudah diputuskan di rancangan ini; pelaksana tidak perlu menebak:

- Agregasi bulanan dihitung langsung dari data 1 menit, bukan dari rata-rata harian.
- Sanity check memakai wind direction dalam derajat sebagai satu fitur (lima input seperti paper).
- Ambang completeness harian: 75% menit valid.
- Fitur kalender memakai day-of-year hari target (t+1).
- Laporan ditulis dalam bahasa Indonesia.
- Tujuh hari kerja 24–30 November; 1 Desember adalah deadline/buffer.

### 13.2 Yang masih diserahkan ke builder

Hal-hal berikut boleh diputuskan sendiri oleh pelaksana notebook, asalkan didokumentasikan:

- Implementasi detail script download (retry, chunk size, format file).
- Metode agregasi presipitasi setelah verifikasi data mentah.
- Ambang interpolasi missing data.
- Grid hyperparameter final (selama masih kecil dan wajar).
- Pilihan plot dan tata letak laporan.
- Penamaan variabel dan struktur folder.
- Apakah XGBoost/stacking ditambahkan sebagai stretch goal.

---

## 14. Definisi Selesai

Tugas dianggap selesai jika:

1. Notebook berjalan dari atas ke bawah tanpa error pada environment yang didokumentasikan.
2. `daily.csv` dan `monthly.csv` dihasilkan oleh pipeline yang dapat dijalankan ulang.
3. Sanity check bulanan menghasilkan tabel metrik dan perbandingan dengan paper.
4. Eksperimen t+1 menghasilkan tabel metrik untuk persistence, LR, SVR, Fine Tree, dan RF.
5. Ada analisis feature importance dan error per musim/level wind speed.
6. `report.md` memuat semua bagian yang diminta, dengan angka yang konsisten dengan notebook.
7. Tidak ada data leakage yang teridentifikasi pada split, scaling, atau fitur.
8. Keterbatasan dan perbedaan dengan paper dijelaskan secara jujur.
