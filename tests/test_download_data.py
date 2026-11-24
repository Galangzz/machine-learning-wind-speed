import pandas as pd

from scripts.download_data import load_month


def test_load_month_drops_duplicate_minutes(tmp_path):
    rows = [
        {"Year": 2019, "DOY": 229, "MST": 1200, "Avg Wind Speed @ 50m [m/s]": 1.0},
        {"Year": 2019, "DOY": 229, "MST": 1201, "Avg Wind Speed @ 50m [m/s]": 2.0},
        {"Year": 2019, "DOY": 229, "MST": 1201, "Avg Wind Speed @ 50m [m/s]": 9.0},
    ]
    path = tmp_path / "nwtc_2019_08.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    raw, n_duplicates = load_month(path)

    assert n_duplicates == 1
    assert len(raw) == 2
    kept = raw.loc[raw["MST"] == 1201, "Avg Wind Speed @ 50m [m/s]"].iloc[0]
    assert kept == 2.0
