# Plan 09 — Reproducibility and Final Verification

> **Plan 09 of 09.** Baca `00-overview.md` dulu. Prasyarat: Plan 08 lulus.

**Goal:** Mengunci environment, memverifikasi ulang seluruh Definition of Done, dan meninggalkan repo dalam keadaan bisa dijalankan ulang dari nol.

**Depends on:** Plan 08.

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md` (opsional: instruksi menjalankan)
- Modify: `main.ipynb` (output eksekusi final)

---

- [ ] **Step 1: Pin the environment**

Run:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project > requirements.txt
```

Expected: `requirements.txt` berisi versi pinned (`==`) untuk seluruh dependency.

- [ ] **Step 2: Run tests**

Run: `uv run pytest -q`

Expected: 6 passed.

- [ ] **Step 3: Execute the notebook from a clean kernel**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected: exit code 0, tidak ada traceback. Notebook tersimpan dengan outputs.

- [ ] **Step 4: Verify the Definition of Done**

Run:

```bash
uv run python - <<'PY'
import pandas as pd

daily = pd.read_csv("data/processed/daily.csv")
monthly = pd.read_csv("data/processed/monthly.csv")
main = pd.read_csv("results/test_metrics.csv")
sanity = pd.read_csv("results/sanity_check_metrics.csv")

print("daily rows:", len(daily))
print("monthly rows:", len(monthly))
print("main models:", main["model"].tolist())
print("sanity models:", sanity["model"].tolist())

assert len(daily) == 1826
assert len(monthly) == 60
assert set(main["model"]) == {"Persistence", "Linear Regression", "SVR", "Fine Tree", "Random Forest"}
assert set(sanity["model"]) == {"Linear Regression", "SVR", "Fine Tree"}
PY
```

Expected: semua assertion lulus.

- [ ] **Step 5: Check repo hygiene**

Run:

```bash
git status --short
git ls-files | grep -E '^(data/raw/|\.venv/|\.ipynb_checkpoints/)' || echo "clean"
```

Expected: tidak ada file `data/raw/`, `.venv/`, atau checkpoint yang ter-track. `git status` hanya menunjukkan perubahan yang memang disengaja.

- [ ] **Step 6: Final commit**

```bash
git add requirements.txt README.md main.ipynb
git commit -m "chore: pin environment and verify reproducibility"
```

---

**Verification gate:** pytest lulus, notebook headless lulus, semua assertion DoD lulus, tidak ada data mentah ter-track. Setelah ini proyek siap diserahkan.
