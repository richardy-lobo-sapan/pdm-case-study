import pandas as pd

from src.data import add_rul, add_test_rul


def _toy() -> pd.DataFrame:
    # engine 1 runs 3 cycles, engine 2 runs 5 cycles
    return pd.DataFrame({"engine": [1, 1, 1, 2, 2, 2, 2, 2], "cycle": [1, 2, 3, 1, 2, 3, 4, 5]})


def test_train_rul_counts_down_to_zero_per_engine():
    out = add_rul(_toy(), cap=125)
    assert out["rul"].tolist() == [2, 1, 0, 4, 3, 2, 1, 0]


def test_cap_clips_only_rul_capped():
    out = add_rul(_toy(), cap=3)
    assert out["rul_capped"].tolist() == [2, 1, 0, 3, 3, 2, 1, 0]
    assert out["rul"].max() == 4  # true RUL untouched


def test_test_rul_adds_offset_from_rul_file():
    true_rul = pd.Series({1: 10, 2: 0})
    out = add_test_rul(_toy(), true_rul, cap=11)
    assert out["rul"].tolist() == [12, 11, 10, 4, 3, 2, 1, 0]
    assert out["rul_capped"].tolist() == [11, 11, 10, 4, 3, 2, 1, 0]
