# Plan 03 — Notebook Skeleton, EDA, and Data Quality

> **Plan 03 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 02 lulus (`daily.csv` 1.826 baris, `monthly.csv` 60 baris).

**Goal:** Mengganti `main.ipynb` placeholder dengan notebook deliverable yang memuat setup, pemuatan data, audit kualitas, dan EDA; semua figure disimpan ke `results/figures/`.

**Depends on:** Plan 02 (dataset hasil agregasi).

**Files:**
- Modify: `main.ipynb`
- Create (generated): `results/figures/*.png`

**Interfaces:**
- Consumes: `data/processed/daily.csv`, `data/processed/monthly.csv`, `data/processed/data_quality.csv`.
- Produces: notebook section 1–3; lima figure; variabel global `daily`, `monthly`, `quality`.

**Konvensi notebook (berlaku untuk seluruh plan berikutnya):**
- Satu cell untuk satu langkah logis.
- Notebook harus jalan dari atas ke bawah tanpa hidden state.
- Semua import diletakkan di section 1. Kalau plan berikutnya butuh import baru, tambahkan ke section 1, bukan di tengah notebook.
- Setiap figure disimpan dengan `dpi=150, bbox_inches="tight"` ke `results/figures/`.
- Tidak ada angka hardcoded yang tidak diverifikasi dari data.

---

- [ ] **Step 1: Replace `main.ipynb` with the project notebook**

Struktur Markdown yang wajib ada:

```markdown
# Daily One-Day-Ahead Wind Speed Forecasting — NWTC M2
## 1. Setup
## 2. Data Loading and Quality
## 3. Exploratory Data Analysis
```

**Cell section 1 — setup:**
- Import `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `json`, `Path`.
- Import scikit-learn: `clone`, `LinearRegression`, `SVR`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GridSearchCV`, `KFold`, `TimeSeriesSplit`, `train_test_split`, `make_pipeline`, `MinMaxScaler`, `mean_absolute_error`, `mean_squared_error`, `r2_score`, `permutation_importance`.
- Konstanta: `DATA_DIR = Path("data/processed")`, `FIG_DIR = Path("results/figures")`, `RESULTS_DIR = Path("results")`, `RANDOM_STATE = 42`.
- `FIG_DIR.mkdir(parents=True, exist_ok=True)`, `RESULTS_DIR.mkdir(parents=True, exist_ok=True)`.
- `pd.set_option("display.max_columns", 100)`, `sns.set_theme(style="whitegrid")`.

**Cell section 2 — load + audit:**
- Load ketiga CSV; `monthly.index = pd.PeriodIndex(monthly.index, freq="M")` setelah `read_csv`.
- Assertion wajib (samakan dengan Plan 02 Step 4):
  ```python
  assert len(daily) == 1826
  assert len(monthly) == 60
  assert len(quality) == 60
  assert daily.index.min() == pd.Timestamp("2019-01-01")
  assert daily.index.max() == pd.Timestamp("2023-12-31")
  assert monthly.index.min() == pd.Period("2019-01", "M")
  assert monthly.index.max() == pd.Period("2023-12", "M")
  assert not daily.index.duplicated().any()
  ```
- Tampilkan `daily.head()`, `monthly.head()`, `quality.head()`.
- Tabel ringkasan: jumlah hari, rentang tanggal, missing per kolom, jumlah bulan, missing per bulan.
- Hitung `valid_wind_speed / expected_minutes` dari `data_quality.csv`, tampilkan 5 bulan terendah.

**Cell section 3 — EDA:**
- Time series harian `wind_speed_50m` 2019–2023 → `fig_daily_wind_speed.png`.
- Time series bulanan `wind_speed_50m` 2019–2023 → `fig_monthly_wind_speed.png`.
- Histogram + KDE `wind_speed_50m` harian → `fig_target_distribution.png`.
- Rata-rata bulanan per bulan kalender (Jan–Dec) → `fig_monthly_climatology.png`.
- Correlation heatmap fitur harian (`wind_speed_50m`, `wind_dir_sin`, `wind_dir_cos`, `temperature_50m`, `relative_humidity`, `pressure`, `precipitation`) → `fig_correlation.png`.
- Tabel korelasi Pearson dengan target `wind_speed_50m`, diurutkan.

- [ ] **Step 2: Execute the notebook headless**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0, tidak ada traceback.

- [ ] **Step 3: Verify figures**

Run: `ls -lh results/figures/`

Expected: minimal 5 PNG yang disebut di Step 1.

- [ ] **Step 4: Commit**

```bash
git add main.ipynb results/figures
git commit -m "feat: add notebook EDA and data quality report"
```

---

**Verification gate:** notebook dieksekusi headless tanpa error; assertion section 2 lulus; minimal 5 figure ada. Jangan lanjut ke Plan 04 sebelum ini lulus.
