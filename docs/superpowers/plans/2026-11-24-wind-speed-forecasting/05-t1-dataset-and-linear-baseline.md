# Plan 05 — t+1 Supervised Dataset, Persistence, and Linear Regression

> **Plan 05 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 04 lulus.

**Goal:** Membangun dataset supervised satu hari ke depan (target hari t+1, fitur sampai akhir hari t), mengaudit leakage, lalu mengevaluasi baseline persistence dan Linear Regression pada test 2023.

**Depends on:** Plan 04 (notebook sudah berjalan; `daily` tersedia).

**Files:**
- Modify: `main.ipynb`
- Create: `results/test_metrics.csv`

**Interfaces:**
- Consumes: `daily` dari notebook section 2.
- Produces:
  - `FEATURES` (16 nama, urutan tetap).
  - `train`, `val`, `test` DataFrame.
  - `results/test_metrics.csv` dengan kolom `model, RMSE, MSE, MAE, R2, RMSE_reduction_pct, MAE_reduction_pct`.
  - `final_models` dict berisi `"Linear Regression"`.
  - `results` list dan fungsi `add_result`.

**Constraints khusus task ini:**
- Fitur hanya data sampai akhir hari t; target `wind_speed_50m` hari t+1.
- Jangan interpolasi `wind_speed_50m` atau target. Interpolasi hanya fitur meteorologi non-wind-speed, maksimal 2 hari, hanya di dalam gap.
- Split kronologis berdasarkan `target_date`: train ≤ 2021-12-31, validation 2022, test ≥ 2023-01-01. Tidak ada shuffling.
- Scaler fit pada train saja; model final (train+validation) memakai pipeline dengan scaler baru.

---

- [ ] **Step 1: Add `## 5. t+1 Dataset and Linear Baseline`**

**Cell 5.1 — feature set (copy persis):**
```python
FEATURES = [
    "wind_speed_lag0", "wind_speed_lag1", "wind_speed_lag2", "wind_speed_lag3",
    "wind_speed_lag4", "wind_speed_lag5", "wind_speed_lag6", "wind_speed_lag7",
    "wind_dir_sin", "wind_dir_cos", "temperature_50m", "pressure",
    "relative_humidity", "precipitation", "doy_sin", "doy_cos",
]
```

**Cell 5.2 — interpolate short feature gaps, then build supervised frame:**

Interpolasi hanya untuk fitur non-wind-speed:
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

**Cell 5.3 — leakage audit (wajib, jangan dilewati):**
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

**Cell 5.4 — chronological split by target date:**
```python
train = sup[sup["target_date"] <= pd.Timestamp("2021-12-31")]
val = sup[(sup["target_date"] >= pd.Timestamp("2022-01-01")) & (sup["target_date"] <= pd.Timestamp("2022-12-31"))]
test = sup[sup["target_date"] >= pd.Timestamp("2023-01-01")]

assert len(train) + len(val) + len(test) == len(sup)
assert train["target_date"].max() < val["target_date"].min()
assert val["target_date"].max() < test["target_date"].min()
```
Cetak `len(train), len(val), len(test)` dan rentang `target_date` masing-masing. Target: train ±1.088, validation 365, test 365 (berkurang jika ada hari hilang; 7 hari pertama dan 1 hari terakhir selalu hilang karena lag/target).

**Cell 5.5 — metrics helper:**
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

**Cell 5.6 — persistence:**
```python
add_result("Persistence", test["target"], test["wind_speed_lag0"])
persistence_rmse = results[0]["RMSE"]
persistence_mae = results[0]["MAE"]
```

**Cell 5.7 — Linear Regression:**
- `lr_scaler = MinMaxScaler(feature_range=(-1, 1))` fit pada `train[FEATURES]`.
- Audit scaling (wajib):
  ```python
  assert np.allclose(lr_scaler.data_min_, train[FEATURES].min().to_numpy())
  assert np.allclose(lr_scaler.data_max_, train[FEATURES].max().to_numpy())
  ```
- Fit `LinearRegression()` pada train yang sudah di-scale (memastikan alur train-only jalan).
- Final model sebagai pipeline: `lr_final = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), LinearRegression())`; fit pada `pd.concat([train, val])[FEATURES]` dan `pd.concat([train, val])["target"]`.
- Evaluasi `lr_final` pada test; `add_result("Linear Regression", test["target"], pred)`.
- Simpan `final_models = {"Linear Regression": lr_final}` untuk Plan 06.

**Cell 5.8 — finalize table:**
```python
metrics_df = pd.DataFrame(results)
metrics_df["RMSE_reduction_pct"] = 100 * (persistence_rmse - metrics_df["RMSE"]) / persistence_rmse
metrics_df["MAE_reduction_pct"] = 100 * (persistence_mae - metrics_df["MAE"]) / persistence_mae
metrics_df = metrics_df.sort_values("RMSE")
metrics_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
metrics_df
```

- [ ] **Step 2: Execute the notebook**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0; `results/test_metrics.csv` berisi 2 baris (Persistence, Linear Regression).

- [ ] **Step 3: Commit**

```bash
git add main.ipynb results/test_metrics.csv
git commit -m "feat: add t+1 dataset and persistence/linear baselines"
```

---

**Verification gate:** semua assertion leakage lulus; `test_metrics.csv` berisi 2 baris; baseline persistence punya `RMSE_reduction_pct = 0`. Jangan lanjut ke Plan 06 sebelum ini lulus.
