# Plan 06 — SVR, Fine Tree, and Random Forest with Tuning

> **Plan 06 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 05 lulus (`train`/`val`/`test`, `FEATURES`, `results`, `add_result`, `final_models` tersedia).

**Goal:** Melatih dan menuning tiga model non-linear (SVR RBF, Fine Tree, Random Forest), memilih hyperparameter lewat time-series cross-validation pada train, lalu mengevaluasi pada test 2023 setelah refit pada train+validation.

**Depends on:** Plan 05.

**Files:**
- Modify: `main.ipynb`
- Create: `results/best_params.json`
- Update: `results/test_metrics.csv` (dari 2 menjadi 5 baris)

**Interfaces:**
- Consumes: `train`, `val`, `test`, `FEATURES`, `results`, `add_result`, `final_models`, `persistence_rmse`, `persistence_mae`.
- Produces: `results/test_metrics.csv` 5 baris; `results/best_params.json`; `final_models` berisi kelima model.

**Constraints khusus task ini:**
- CV untuk tuning memakai `TimeSeriesSplit(n_splits=3)` pada train. Tidak ada shuffling.
- Refit final memakai `clone(best_estimator_)` pada train+validation; pipeline scaler otomatis di-fit ulang pada data gabungan (bukan memakai scaler train).
- `random_state=42` di semua model dan search.
- Scope model hanya SVR, Fine Tree, Random Forest. Tidak ada XGBoost/stacking/LSTM.

---

- [ ] **Step 1: Add `## 6. Nonlinear Models and Tuning`**

**Cell 6.1 — shared setup:**
```python
tscv = TimeSeriesSplit(n_splits=3)
X_train, y_train = train[FEATURES], train["target"]
X_val, y_val = val[FEATURES], val["target"]
X_test, y_test = test[FEATURES], test["target"]
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
```

**Cell 6.2 — SVR:**
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

**Cell 6.3 — Fine Tree:**
```python
tree_grid = {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 4, 8]}
tree_search = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE), tree_grid, cv=tscv,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
tree_search.fit(X_train, y_train)
```

**Cell 6.4 — Random Forest:**
```python
rf_grid = {
    "n_estimators": [300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0],
}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1,
)
rf_search.fit(X_train, y_train)
```
Catatan: `n_jobs=1` di GridSearchCV karena estimator RF sudah paralel; menghindari oversubscription.

**Cell 6.5 — validation + refit + test untuk setiap model:**
- Evaluasi `best_estimator_` pada validation; catat `val_rmse`.
- `final = clone(search.best_estimator_)`, fit pada `X_trainval, y_trainval`.
- Prediksi test; `add_result("SVR", y_test, pred)` dan seterusnya.
- Tambahkan ke `final_models`:
  ```python
  final_models.update({"SVR": svr_final, "Fine Tree": tree_final, "Random Forest": rf_final})
  ```

**Cell 6.6 — best params + final table:**
- Simpan:
  ```python
  best = {
      "SVR": {"params": svr_search.best_params_, "val_rmse": svr_val_rmse},
      "Fine Tree": {"params": tree_search.best_params_, "val_rmse": tree_val_rmse},
      "Random Forest": {"params": rf_search.best_params_, "val_rmse": rf_val_rmse},
  }
  with open(RESULTS_DIR / "best_params.json", "w") as f:
      json.dump(best, f, indent=2)
  ```
- Hitung ulang reduction vs persistence, tulis `results/test_metrics.csv` (5 baris), tampilkan diurutkan berdasarkan RMSE.
- Sanity guard:
  ```python
  assert np.isfinite(metrics_df[["RMSE", "MSE", "MAE", "R2"]].to_numpy()).all()
  assert metrics_df.loc[metrics_df["model"] == "Random Forest", "RMSE"].iloc[0] < 100
  ```

- [ ] **Step 2: Execute the notebook**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0; `results/test_metrics.csv` berisi 5 baris; `results/best_params.json` ada dan valid.

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/test_metrics.csv results/best_params.json
git commit -m "feat: tune and evaluate SVR, fine tree, and random forest"
```

---

**Verification gate:** `test_metrics.csv` berisi tepat 5 model (`Persistence`, `Linear Regression`, `SVR`, `Fine Tree`, `Random Forest`); semua metrik finite; `best_params.json` memuat tiga model. Jangan lanjut ke Plan 07 sebelum ini lulus.
