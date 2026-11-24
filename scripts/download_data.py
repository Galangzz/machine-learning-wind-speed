"""Unduh export CSV MIDC NWTC M2 per bulan dan agregasi ke harian/bulanan.

Jalankan sebagai module:

    uv run python -m scripts.download_data
"""

from __future__ import annotations

import calendar
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from scripts.preprocess import aggregate_daily, aggregate_monthly, parse_midc

BASE_URL = "https://midcdmz.nlr.gov/apps/data_api.pl"
SITE = "NWTC"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

START = (2019, 1)
END = (2023, 12)

MINUTES_PER_DAY = 1440
MIN_ROW_FRACTION = 0.99
RETRY_BACKOFF_SECONDS = (5, 10)


def month_path(year: int, month: int, raw_dir: Path = RAW_DIR) -> Path:
    return Path(raw_dir) / f"nwtc_{year}_{month:02d}.csv"


def count_data_rows(path: Path) -> int:
    with open(path, "rb") as handle:
        lines = sum(1 for _ in handle)
    return max(lines - 1, 0)


def fetch_month(year: int, month: int, raw_dir: Path = RAW_DIR, retries: int = 3) -> Path:
    """Unduh satu bulan; file yang sudah lengkap dipakai ulang (resume)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = month_path(year, month, raw_dir)

    days = calendar.monthrange(year, month)[1]
    expected_rows = days * MINUTES_PER_DAY

    if path.exists():
        rows = count_data_rows(path)
        if rows >= MIN_ROW_FRACTION * expected_rows:
            print(f"{year}-{month:02d}: memakai file lokal ({rows} baris)")
            return path
        print(f"{year}-{month:02d}: file lokal terpotong ({rows} baris), unduh ulang")

    params = {
        "site": SITE,
        "begin": f"{year}{month:02d}01",
        "end": f"{year}{month:02d}{days:02d}",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        if attempt:
            backoff = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
            print(f"{year}-{month:02d}: percobaan ulang {attempt + 1}/{retries} setelah {backoff}s")
            time.sleep(backoff)
        try:
            response = requests.get(BASE_URL, params=params, timeout=180)
            response.raise_for_status()

            part = path.with_name(path.name + ".part")
            part.write_bytes(response.content)
            rows = count_data_rows(part)
            if rows < MIN_ROW_FRACTION * expected_rows:
                part.unlink(missing_ok=True)
                raise ValueError(f"respons terpotong: {rows} baris, ekspektasi {expected_rows}")
            part.replace(path)
            print(f"{year}-{month:02d}: {rows} baris diunduh ({response.content.__len__() / 1e6:.1f} MB)")
            return path
        except Exception as exc:  # noqa: BLE001 - semua kegagalan jaringan/validasi di-retry
            last_error = exc
            print(f"{year}-{month:02d}: gagal ({exc})")

    raise RuntimeError(f"gagal mengunduh {year}-{month:02d} setelah {retries} percobaan: {last_error}")


def load_month(path: Path) -> tuple[pd.DataFrame, int]:
    """Baca CSV mentah; menit yang terduplikasi di sumber diambil kemunculan pertama."""
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]

    keys = ["Year", "DOY", "MST"]
    duplicates = raw.duplicated(subset=keys, keep="first")
    n_duplicates = int(duplicates.sum())
    if n_duplicates:
        sample = raw.loc[duplicates, keys].head(3).to_dict("records")
        print(f"peringatan: {Path(path).name} memuat {n_duplicates} menit duplikat di sumber; dipakai kemunculan pertama, contoh {sample}")
        raw = raw.loc[~duplicates]

    return raw, n_duplicates


def quality_record(year: int, month: int, df: pd.DataFrame, n_duplicates: int = 0) -> dict:
    days = calendar.monthrange(year, month)[1]
    expected_minutes = days * MINUTES_PER_DAY
    return {
        "year": year,
        "month": month,
        "expected_minutes": expected_minutes,
        "n_rows": len(df),
        "n_missing_minutes": expected_minutes - len(df),
        "n_duplicate_minutes": n_duplicates,
        "valid_wind_speed": int(df["wind_speed_50m"].notna().sum()),
        "valid_wind_dir": int(df["wind_dir_50m"].notna().sum()),
        "valid_temperature": int(df["temperature_50m"].notna().sum()),
        "valid_humidity": int(df["relative_humidity"].notna().sum()),
        "valid_pressure": int(df["pressure"].notna().sum()),
        "valid_precipitation": int(df["precipitation"].notna().sum()),
    }


def main(start: tuple[int, int] = START, end: tuple[int, int] = END) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    quality_rows: list[dict] = []

    started = time.time()
    for year in range(start[0], end[0] + 1):
        for month in range(1, 13):
            if (year, month) < start or (year, month) > end:
                continue
            path = fetch_month(year, month)
            raw, n_duplicates = load_month(path)
            df = parse_midc(raw)
            quality_rows.append(quality_record(year, month, df, n_duplicates))
            daily_parts.append(aggregate_daily(df))
            monthly_parts.append(aggregate_monthly(df))
            print(f"{year}-{month:02d}: {len(df)} menit valid-diparse (total {time.time() - started:.0f}s)")

    daily = pd.concat(daily_parts).sort_index()
    daily = daily.reindex(pd.date_range("2019-01-01", "2023-12-31", freq="D"))
    daily.index.name = "date"

    monthly = pd.concat(monthly_parts).sort_index()
    monthly = monthly.reindex(pd.period_range("2019-01", "2023-12", freq="M"))
    monthly.index.name = "month"

    quality = pd.DataFrame(quality_rows)

    daily.to_csv(PROCESSED_DIR / "daily.csv", float_format="%.6f")
    monthly.to_csv(PROCESSED_DIR / "monthly.csv", float_format="%.6f")
    quality.to_csv(PROCESSED_DIR / "data_quality.csv", index=False)

    print(f"daily: {daily.shape}, monthly: {monthly.shape}, quality: {quality.shape}")
    print(f"selesai dalam {time.time() - started:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("pemakaian: uv run python -m scripts.download_data", file=sys.stderr)
        raise SystemExit(2)
    main()
