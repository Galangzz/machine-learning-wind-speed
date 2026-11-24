# Plan 07 — Feature Importance and Error Analysis

> **Plan 07 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 06 lulus (lima model final dan metrik test tersedia).

**Goal:** Menjelaskan mengapa model berperforma demikian: fitur mana yang dominan, di musim apa error terbesar, dan pada level kecepatan angin apa model gagal.

**Depends on:** Plan 06 (`final_models`, `test`, `FEATURES`, `results/test_metrics.csv`).

**Files:**
- Modify: `main.ipynb`
- Create: `results/feature_importance.csv`, `results/error_by_season.csv`, `results/error_by_level.csv`
- Create: enam PNG di `results/figures/`

**Interfaces:**
- Consumes: `final_models`, `test`, `FEATURES`, `results/test_metrics.csv`.
- Produces: tiga CSV tabel dan enam figure.

**Constraints khusus task ini:**
- `permutation_importance` memakai `n_repeats=30`, `random_state=42`, `scoring="neg_root_mean_squared_error"`.
- Model terbaik = RMSE terendah pada `results/test_metrics.csv`, **tetapi persistence tidak boleh dipilih sebagai "model terbaik"**; jika persistence yang terendah, ambil model ML terbaik dan sebutkan bahwa persistence menang.
- Koefisien Linear Regression hanya diinterpretasikan setelah scaling; tulis catatan itu di Markdown.
- Semua tabel disimpan ke `results/`, semua figure ke `results/figures/` dengan `dpi=150, bbox_inches="tight"`.

---

- [ ] **Step 1: Add `## 7. Feature Importance and Error Analysis`**

**Cell 7.1 — importance table:**
- `rf = final_models["Random Forest"]`; impurity importance `rf.feature_importances_`.
- Tentukan model ML terbaik dari `metrics_df` (RMSE terendah, exclude `Persistence`).
- `permutation_importance(best_ml_model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1)`.
- Koefisien LR: `final_models["Linear Regression"].named_steps["linearregression"].coef_`.
- Gabungkan ke DataFrame `feature, rf_impurity, permutation_mean, permutation_std, lr_coef`; urutkan berdasarkan `permutation_mean` menurun; simpan `results/feature_importance.csv`.
- Plot top-10 permutation importance → `fig_feature_importance.png`.

**Cell 7.2 — error by season:**
- Mapping berdasarkan `test["target_date"].dt.month`: DJF = 12,1,2; MAM = 3,4,5; JJA = 6,7,8; SON = 9,10,11.
- Untuk model ML terbaik dan persistence, hitung RMSE dan MAE per musim.
- Simpan `results/error_by_season.csv` dengan kolom `season, model, RMSE, MAE, n`.
- Grouped bar chart → `fig_error_by_season.png`.

**Cell 7.3 — error by wind speed level:**
- `test["level"] = pd.qcut(test["target"], 3, labels=["low", "mid", "high"], duplicates="drop")`.
- Hitung RMSE/MAE per level untuk model ML terbaik dan persistence.
- Simpan `results/error_by_level.csv` dengan kolom `level, model, RMSE, MAE, n`.
- Plot → `fig_error_by_level.png`.

**Cell 7.4 — prediction plots:**
- Time series actual vs model terbaik vs persistence pada test 2023 → `fig_test_timeseries.png`.
- Scatter predicted vs actual model terbaik dengan garis 1:1 → `fig_test_scatter.png`.
- Histogram residual model terbaik → `fig_residuals.png`.

**Cell 7.5 — ringkasan:**
- Tampilkan tabel akhir `results/test_metrics.csv` diurutkan berdasarkan RMSE.
- Cetak model terbaik dan RMSE reduction terhadap persistence.
- Cetak top-5 fitur berdasarkan permutation importance.
- Cetak satu paragraf singkat: apakah model ML mengalahkan persistence, dan fitur apa yang dominan.

- [ ] **Step 2: Execute the notebook**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0; tiga CSV dan enam PNG baru ada.

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/feature_importance.csv results/error_by_season.csv results/error_by_level.csv results/figures
git commit -m "feat: add feature importance and error analysis"
```

---

**Verification gate:** tiga CSV tidak kosong dan memuat model/persistence; enam PNG ada; notebook jalan headless. Jangan lanjut ke Plan 08 sebelum ini lulus.
