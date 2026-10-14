import numpy as np
import pandas as pd

from src.anomaly import engine_summary, persistent_alarm
from src.data import engine_folds


def test_alarm_needs_consecutive_cycles_within_one_engine():
    above = pd.Series([True, True, False, True, True, True, True, True, False, False])
    engine = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    alarm = persistent_alarm(above, engine, persistence=3)
    # engine 1: run of 3 completes at row 5; engine 2: its 2 Trues can't borrow engine 1's run
    assert alarm.tolist() == [False, False, False, False, False, True, False, False, False, False]


def test_engine_summary_lead_time_and_false_alarm():
    # engine 1: alarm at RUL 200 (healthy -> false alarm) and at RUL 40/39 (first in window = 40)
    # engine 2: never alarms -> missed
    scored = pd.DataFrame({
        "engine": [1, 1, 1, 1, 2, 2],
        "cycle": [40, 50, 210, 211, 40, 50],
        "rul": [200, 190, 40, 39, 160, 0],
        "m_alarm": [True, False, True, True, False, False],
    })
    e = engine_summary(scored, "m", baseline_cycles=30, healthy_rul_min=150)
    assert e.loc[1, "false_alarm"] and not e.loc[2, "false_alarm"]
    assert e.loc[1, "lead_time"] == 40
    assert np.isnan(e.loc[2, "lead_time"])


def test_engine_folds_never_share_an_engine():
    df = pd.DataFrame({"engine": np.repeat(np.arange(1, 21), 7)})
    seen = []
    for tr, va in engine_folds(df, n_splits=5):
        assert set(df.engine.iloc[tr]).isdisjoint(df.engine.iloc[va])
        seen.extend(df.engine.iloc[va].unique())
    assert sorted(seen) == list(range(1, 21))  # every engine validated exactly once
