import numpy as np
import pandas as pd

from src.features import add_rolling_features
from src.rul import conformal_margin, coverage, coverage_by_band, nasa_score


def test_nasa_score_penalizes_late_more_than_early():
    late, early = nasa_score([50], [60])[0], nasa_score([50], [40])[0]
    assert np.isclose(late, np.exp(1) - 1) and np.isclose(early, np.exp(10 / 13) - 1)
    assert late > early
    assert nasa_score([50], [50])[0] == 0


def test_conformal_margin_fixes_too_narrow_ranges():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=5000), rng.normal(size=5000)
    lo, hi = np.full(5000, -0.5), np.full(5000, 0.5)  # covers only ~38%
    m = conformal_margin(y_cal, lo, hi, target=0.9)
    assert 0.88 <= coverage(y_new, lo - m, hi + m) <= 0.92


def test_coverage_by_band_splits_misses_by_distance_to_failure():
    preds = pd.DataFrame({"rul": [5, 8, 30, 40], "lo": [0, 9, 20, 20], "hi": [20, 20, 35, 35]})
    b = coverage_by_band(preds, [0, 10, 50])
    assert b.index.tolist() == ["(-0.001, 10.0]", "(10.0, 50.0]"]
    assert b["coverage"].tolist() == [0.5, 0.5]
    assert b["below_lower_bound"].tolist() == [0.5, 0.0]  # rul 8 < lo 9 near failure; rul 40 > hi far from it


def test_features_never_look_ahead():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"engine": 1, "cycle": np.arange(1, 41), "s2": rng.normal(size=40), "s3": rng.normal(size=40)})
    before, cols = add_rolling_features(df, ["s2", "s3"], [5, 10], baseline_cycles=30)
    changed = df.copy()
    changed.loc[changed.index[-1], ["s2", "s3"]] = 999.0  # alter only the last cycle
    after, _ = add_rolling_features(changed, ["s2", "s3"], [5, 10], baseline_cycles=30)
    pd.testing.assert_frame_equal(before[cols].iloc[:-1], after[cols].iloc[:-1])
    assert not before[cols].iloc[-1].equals(after[cols].iloc[-1])


def test_baseline_deviation_uses_only_past_cycles():
    df = pd.DataFrame({"engine": 1, "cycle": [1, 2, 3], "s2": [1.0, 3.0, 5.0]})
    out, _ = add_rolling_features(df, ["s2"], [2], baseline_cycles=2)
    # baseline at cycle 1 = 1; cycle 2 = mean(1,3) = 2; cycle 3 frozen at 2
    assert out["s2_dev"].tolist() == [0.0, 1.0, 3.0]
