import numpy as np
import pandas as pd

from src.data import SENSORS, SETTINGS, add_rul, add_test_rul, normalize_by_regime


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


def test_regime_normalization_removes_operating_condition_offsets():
    # two operating regimes; s2 reads 100 higher in regime B for the same health; s1 is constant per regime
    rng = np.random.default_rng(0)
    regime_b = np.tile([False, True], 50)
    df = pd.DataFrame(0.0, index=range(100), columns=SETTINGS + SENSORS).assign(engine=1, cycle=range(1, 101))
    df["setting_1"] = np.where(regime_b, 40.0, 0.0)
    df["s2"] = np.where(regime_b, 100.0, 0.0) + rng.normal(size=100)
    df["s1"] = np.where(regime_b, 7.0, 3.0)
    train, test = normalize_by_regime(df, df.iloc[:10], n_regimes=2, seed=0)
    assert abs(train.loc[regime_b, "s2"].mean()) < 1e-9 and abs(train.loc[~regime_b, "s2"].mean()) < 1e-9
    assert (train["s1"] == 0).all()  # constant within each regime: no signal, and no division by zero
    pd.testing.assert_series_equal(test["s2"], train["s2"].iloc[:10])  # test uses train's regimes and stats
    assert normalize_by_regime(df, df, n_regimes=1, seed=0)[0] is df  # single regime (FD001): untouched


def test_test_rul_adds_offset_from_rul_file():
    true_rul = pd.Series({1: 10, 2: 0})
    out = add_test_rul(_toy(), true_rul, cap=11)
    assert out["rul"].tolist() == [12, 11, 10, 4, 3, 2, 1, 0]
    assert out["rul_capped"].tolist() == [11, 11, 10, 4, 3, 2, 1, 0]
