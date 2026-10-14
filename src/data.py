"""Load NASA C-MAPSS files and add remaining-life (RUL) labels."""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["engine", "cycle", *SETTINGS, *SENSORS]


def load_config(path: Path = ROOT / "configs" / "config.yaml") -> dict:
    """Read the project config (paths, RUL cap, seeds)."""
    return yaml.safe_load(path.read_text())


def _read_table(path: Path) -> pd.DataFrame:
    """Raw files are space-separated with trailing spaces and no header."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_fd(subset: str = "FD001", raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (train, test, true_rul) for one C-MAPSS subset.

    true_rul is indexed by engine id: cycles left after the last test row.
    """
    raw_dir = raw_dir or ROOT / load_config()["paths"]["raw"]
    train = _read_table(raw_dir / f"train_{subset}.txt")
    test = _read_table(raw_dir / f"test_{subset}.txt")
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="engine")  # row i = engine i
    return train, test, rul.rename("rul_at_end")


def add_rul(train: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add true RUL and capped RUL to run-to-failure data.

    Each train engine fails at its last cycle, so RUL = last cycle - current cycle.
    `rul` keeps the truth (needed for cost simulation); `rul_capped` is the training target.
    """
    last = train.groupby("engine")["cycle"].transform("max")
    out = train.assign(rul=last - train["cycle"])
    return out.assign(rul_capped=out["rul"].clip(upper=cap))


def add_test_rul(test: pd.DataFrame, true_rul: pd.Series, cap: int) -> pd.DataFrame:
    """Add RUL labels to test data, which stops before failure.

    true_rul gives cycles left at each engine's last row, so for any row:
    RUL = true_rul[engine] + (last cycle - current cycle).
    """
    last = test.groupby("engine")["cycle"].transform("max")
    out = test.assign(rul=test["engine"].map(true_rul) + last - test["cycle"])
    return out.assign(rul_capped=out["rul"].clip(upper=cap))


def near_constant_columns(train: pd.DataFrame, std_threshold: float) -> list[str]:
    """Settings/sensors whose std is below threshold: no health signal (see notebooks/01_eda)."""
    std = train[SETTINGS + SENSORS].std()
    return std.index[std < std_threshold].tolist()


def engine_folds(df: pd.DataFrame, n_splits: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) row positions with every engine wholly on one side.

    Rows of one engine are near-duplicates; splitting by row would leak answers.
    """
    from sklearn.model_selection import GroupKFold  # imported here so the API image needs no scikit-learn

    yield from GroupKFold(n_splits=n_splits).split(df, groups=df["engine"])
