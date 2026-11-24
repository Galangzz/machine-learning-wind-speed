# Plan 04 — Monthly Sanity Check (Replication)

> **Plan 04 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 03 lulus (notebook section 1–3 jalan).

**Goal:** Mereplikasi skenario paper Altork (2025) secara ringan pada 60 observasi bulanan dengan tiga model paper, sebagai sanity check bahwa parsing dan agregasi menghasilkan pola yang masuk akal.

**Depends on:** Plan 03 (`monthly` DataFrame dan folder `results/`).

**Files:**
- Modify: `main.ipynb`
- Create: `results/sanity_check_metrics.csv`

**Interfaces:**
- Consumes: `monthly` (60 baris) dari notebook section 2.
- Produces: `results/sanity_check_metrics.csv` dengan kolom `model, RMSE, MSE, MAE, R2` dan nama model persis `Linear Regression`, `SVR`, `Fine Tree`.

**Constraints khusus task ini:**
- Ini **sanity check**, bukan eksperimen utama. Split acak 70/15/15 dengan `random_state=42` memang disengaja dan meniru paper; aturan split kronologis tidak berlaku di sini.
- Fitur persis lima seperti paper: `pressure`, `temperature_50m`, `wind_dir_deg`, `relative_humidity`, `precipitation`. Arah angin satu kolom derajat, bukan sin/cos.
- Scaler fit pada train saja.
- Angka paper tidak harus identik; jangan tuning agar cocok.

---

- [ ] **Step 1: Add `## 4. Monthly Sanity Check (Replication)` to the notebook**

**Cell 4.1 — dataset dan split:**
```python
sanity_features = ["pressure", "temperature_50m", "wind_dir_deg", "relative_humidity", "precipitation"]
sanity_target = "wind_speed_50m"

sanity = monthly.dropna(subset=sanity_features + [sanity_target]).copy()
print("sanity observations:", len(sanity))

X = sanity[sanity_features]
y = sanity[sanity_target]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.50, random_state=RANDOM_STATE
)
print("train/val/test:", len(X_train), len(X_val), len(X_test))
```
Target ukuran: 42/9/9 dari 60 observasi. Kalau jumlah observasi bukan 60, laporkan apa adanya dan jelaskan di laporan.

**Cell 4.2 — models:**
- Linear Regression default.
- SVR RBF: `C ∈ {1, 10, 100}`, `gamma ∈ {"scale", 0.1}`, `epsilon ∈ {0.01, 0.1}`; pipeline `MinMaxScaler(feature_range=(-1, 1))` + `SVR(kernel="rbf")`; `GridSearchCV(..., cv=KFold(3, shuffle=True, random_state=42), scoring="neg_root_mean_squared_error")` pada train.
- Fine Tree: `DecisionTreeRegressor(random_state=RANDOM_STATE)` dengan grid `max_depth ∈ {None, 3, 5, 10}`, `min_samples_leaf ∈ {1, 2, 4}`; pencarian yang sama.
- Untuk setiap model: fit pada train, evaluasi validation, lalu refit pada train+validation dengan scaler baru (pipeline baru), evaluasi test.
- Hitung RMSE, MSE, MAE, R² dengan `mean_squared_error`, `mean_absolute_error`, `r2_score`.

**Cell 4.3 — reporting:**
- Tabel metrik test ketiga model, diurutkan berdasarkan RMSE.
- Tabel pembanding berisi angka paper:
  - LR: RMSE 0,29555 / MSE 0,087352 / MAE 0,18061 / R² 0,97
  - SVM: RMSE 0,32275 / R² 0,96
  - Fine Tree: RMSE 0,44042 / R² 0,93
- Catatan eksplisit di Markdown: angka tidak harus identik karena paper tidak merinci hyperparameter, split, dan agregasi.
- Sanity guard:
  ```python
  assert np.isfinite(sanity_results[["RMSE", "MSE", "MAE", "R2"]].to_numpy()).all()
  lr_r2 = sanity_results.loc[sanity_results["model"] == "Linear Regression", "R2"].iloc[0]
  assert 0.5 < lr_r2 < 0.999
  ```
  Jika di luar rentang, selidiki leakage atau parsing sebelum lanjut.
- Simpan `sanity_results.to_csv(RESULTS_DIR / "sanity_check_metrics.csv", index=False)`.

- [ ] **Step 2: Execute the notebook**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0; `results/sanity_check_metrics.csv` berisi 3 baris.

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/sanity_check_metrics.csv
git commit -m "feat: add monthly replication sanity check"
```

---

**Verification gate:** CSV berisi tepat 3 baris dengan nama model `Linear Regression`, `SVR`, `Fine Tree`; semua metrik finite; R² LR di antara 0,5 dan 0,999. Jangan lanjut ke Plan 05 sebelum ini lulus.
