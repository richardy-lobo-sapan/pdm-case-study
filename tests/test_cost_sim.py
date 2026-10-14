import numpy as np
import pandas as pd

from src.cost_sim import engine_costs, evaluate, event_costs, trigger_cycles

C = {"unplanned_failure_cost": 100_000, "planned_replacement_cost": 20_000, "downtime_cost_per_cycle": 5_000,
     "unplanned_downtime_cycles": 10, "planned_downtime_cycles": 2, "wasted_life_cost_per_cycle": 100}


def test_event_costs_include_downtime():
    assert event_costs(C) == (150_000, 30_000)


def test_engine_costs_fail_vs_replace():
    life = np.array([200, 200, 200, 200])
    replace_at = np.array([np.inf, 200, 250, 150])  # never, exactly at failure, too late, in time
    assert engine_costs(replace_at, life, C).tolist() == [150_000, 150_000, 150_000, 30_000 + 50 * 100]


def test_trigger_uses_first_crossing_even_if_signal_bounces_back():
    preds = pd.DataFrame({"engine": 1, "cycle": [1, 2, 3, 4], "lo": [100.0, 30.0, 60.0, 10.0]})
    t = trigger_cycles(preds, "lo", np.array([5, 40, 200]))
    assert t.loc[1].tolist() == [np.inf, 2, 1]


def test_lead_time_can_turn_a_warning_into_a_failure():
    # two identical engines, life 100, perfect lower bound (lo = true RUL); threshold 3 fires at cycle 97
    cycles = np.arange(1, 101)
    preds = pd.DataFrame({"engine": np.repeat([1, 2], 100), "cycle": np.tile(cycles, 2),
                          "lo": np.tile(100.0 - cycles, 2)})
    lives = pd.Series({1: 100, 2: 100}).rename_axis("engine")
    tables = {"predictive_lo": trigger_cycles(preds, "lo", np.array([3]))}
    on_time = evaluate(lives, tables, C, lead=2, n_folds=2)  # replaced at 99
    late = evaluate(lives, tables, C, lead=5, n_folds=2)     # would be 102: fails at 100 first
    assert not on_time["predictive_lo_failed"].any() and on_time["predictive_lo_wasted"].tolist() == [1, 1]
    assert late["predictive_lo_failed"].all() and late["predictive_lo_cost"].tolist() == [150_000, 150_000]


def test_settings_are_chosen_without_the_scored_engine():
    # 99 engines live 100 cycles, engine 1 lives 10. With a huge failure cost, any fit set that
    # contains engine 1 picks N=9; the fold that scores engine 1 never saw it, so it picks N=99.
    c = {**C, "unplanned_failure_cost": 10_000_000}
    lives = pd.Series(100, index=pd.RangeIndex(1, 101, name="engine"))
    lives.loc[1] = 10
    n = np.arange(1, 101)
    tables = {"fixed": pd.DataFrame(np.tile(n, (100, 1)).astype(float), index=lives.index, columns=n)}
    rows = evaluate(lives, tables, c, lead=0, n_folds=5)
    assert rows.loc[1, "fixed_setting"] == 99 and rows.loc[1, "fixed_failed"]
    assert (rows["fixed_setting"] == 9).sum() == 80  # the other folds all saw engine 1
