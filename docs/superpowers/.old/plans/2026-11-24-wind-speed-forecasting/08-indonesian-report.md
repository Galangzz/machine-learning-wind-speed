# Plan 08 — Indonesian Markdown Report

> **Plan 08 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 07 lulus (semua CSV dan figure final tersedia).

**Goal:** Menulis `report.md` berbahasa Indonesia yang melaporkan replikasi dan eksperimen t+1 secara jujur, dengan setiap angka dapat dilacak ke `results/`.

**Depends on:** Plan 07.

**Files:**
- Create: `report.md`

**Interfaces:**
- Consumes: `results/sanity_check_metrics.csv`, `results/test_metrics.csv`, `results/feature_importance.csv`, `results/error_by_season.csv`, `results/error_by_level.csv`, `results/best_params.json`, `results/figures/*.png`, output notebook.
- Produces: `report.md`.

**Aturan angka (tidak boleh dilanggar):**
- Setiap angka di `report.md` harus berasal dari `results/*.csv`, `results/best_params.json`, atau output notebook.
- Jangan mengarang, membulatkan dari ingatan, atau menyalin angka paper ke tabel hasil.
- Angka paper hanya boleh muncul di tabel pembanding sanity check, diberi label jelas sebagai "angka paper".
- Jika hasil bertentangan dengan ekspektasi, laporkan apa adanya.

---

- [ ] **Step 1: Write `report.md` with this structure**

```markdown
# Peramalan Kecepatan Angin Harian Satu Hari ke Depan: Replikasi dan Ekstensi Perbandingan Model Machine Learning

## Abstrak
## 1. Pendahuluan
## 2. Tinjauan Pustaka
## 3. Data dan Preprocessing
## 4. Metodologi
## 5. Hasil Sanity Check Replikasi
## 6. Hasil Peramalan t+1
## 7. Diskusi
## 8. Keterbatasan
## 9. Kesimpulan
## Referensi
## Lampiran
```

Konten wajib:
- **Abstrak:** 150–250 kata; tujuan, data, model, temuan utama.
- **Pendahuluan:** latar belakang energi angin, paper Altork (2025), gap same-period vs forecasting, pertanyaan riset.
- **Tinjauan Pustaka:** tiga model paper, Random Forest (Breiman 2001), data sirkular (Fisher 1993), peramalan time series (Hyndman & Athanasopoulos 2021).
- **Data dan Preprocessing:** sumber MIDC/NWTC M2, periode 2019–2023, agregasi 1 menit → harian/bulanan, aturan completeness 75%, penanganan presipitasi dan arah angin, missing data.
- **Metodologi:** definisi t+1, 16 fitur, split kronologis, scaling, hyperparameter tuning, metrik, persistence baseline.
- **Hasil Sanity Check:** tabel 3 model + angka paper; jelaskan bahwa angka tidak harus identik dan mengapa.
- **Hasil t+1:** tabel 5 model (persistence + 4 model) dengan RMSE/MSE/MAE/R² dan reduction vs persistence; feature importance; error per musim dan level; gambar.
- **Diskusi:** mengapa R² turun dibanding paper; apakah model non-linear mengungguli LR; apakah ML mengalahkan persistence; fitur dominan; implikasi praktis.
- **Keterbatasan:** satu lokasi, 5 tahun, resolusi harian, tanpa NWP, tanpa interval ketidakpastian, presipitasi jarang, asumsi stationarity.
- **Kesimpulan:** jawab pertanyaan riset secara eksplisit.
- **Referensi:** Altork (2025), Jager & Andreas (1996), Breiman (2001), Fisher (1993), Hyndman & Athanasopoulos (2021).
- **Lampiran:** hyperparameter final dari `best_params.json`, versi library, seed.

Aturan tambahan:
- Gunakan gambar dengan path relatif, contoh `![Feature importance](results/figures/fig_feature_importance.png)`.
- Jangan mengklaim paper salah; posisikan sebagai ekstensi dan koreksi protokol.
- Sebutkan bahwa sanity check memakai split acak (sesuai paper), sedangkan eksperimen utama memakai split kronologis.

- [ ] **Step 2: Consistency check**

Run:

```bash
uv run python - <<'PY'
import json
import pandas as pd

for path in [
    "results/sanity_check_metrics.csv",
    "results/test_metrics.csv",
    "results/feature_importance.csv",
    "results/error_by_season.csv",
    "results/error_by_level.csv",
]:
    df = pd.read_csv(path)
    print(path, df.shape)
    assert not df.empty

with open("results/best_params.json") as f:
    print(json.load(f))
PY
```

Expected: semua file ada dan tidak kosong.

- [ ] **Step 3: Commit**

```bash
git add report.md
git commit -m "docs: add Indonesian analysis report"
```

---

**Verification gate:** `report.md` memuat semua section; setiap tabel hasil bisa dicocokkan dengan CSV; tidak ada angka tanpa sumber. Jangan lanjut ke Plan 09 sebelum ini lulus.
