import numpy as np
import pandas as pd

from src.cost_sim import engine_costs, evaluate, event_costs, nested_savings_range, sensitivity, trigger_cycles

C = {"unplanned_failure_cost": 100_000, "planned_replacement_cost": 20_000, "downtime_cost_per_cycle": 5_000,
     "unplanned_downtime_cycles": 10, "planned_downtime_cycles": 2, "wasted_life_cost_per_cycle": 100,
     "maintenance_lead_cycles": 0}


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


def test_nested_range_collapses_when_every_engine_is_the_same():
    # identical engines: any resample picks the same settings and costs, so the range is one point,
    # equal to the plain (non-bootstrap) savings
    lives = pd.Series(100, index=pd.RangeIndex(1, 21, name="engine"))
    cycles = np.arange(1, 101)
    preds = pd.DataFrame({"engine": np.repeat(lives.index, 100), "cycle": np.tile(cycles, 20), "lo": np.tile(100.0 - cycles, 20)})
    tables = {"fixed": pd.DataFrame(np.tile(np.arange(1, 121), (20, 1)).astype(float), index=lives.index, columns=np.arange(1, 121)),
              "predictive_lo": trigger_cycles(preds, "lo", np.arange(0, 11))}
    rows = evaluate(lives, tables, C, lead=0, n_folds=4)
    point = 1 - rows["predictive_lo_cost"].mean() / rows["fixed_cost"].mean()
    lo, hi = nested_savings_range(lives, tables, C, lead=0, n_folds=4, n_boot=20, seed=0, alternative="fixed")
    assert np.isclose(lo, point) and np.isclose(hi, point)


def test_ratio_sweep_scales_the_whole_failure_cost_including_downtime():
    # 20 engines living 81-119 cycles, perfect lower bound. At ratio 1 a breakdown costs exactly a planned swap,
    # so predicting it can't beat running to failure (replacing early only adds wasted life). At ratio 5 it beats
    # a fixed schedule, which must replace before the shortest life and so wastes life on the others.
    lives = pd.Series(np.arange(81, 121, 2), index=pd.RangeIndex(1, 21, name="engine"))
    preds = pd.concat([pd.DataFrame({"engine": e, "cycle": np.arange(1, L + 1), "lo": L - np.arange(1, L + 1.0)})
                       for e, L in lives.items()])
    n = np.arange(1, 121)
    tables = {"run_to_failure": pd.DataFrame(np.inf, index=lives.index, columns=[0]),
              "fixed": pd.DataFrame(np.tile(n, (20, 1)).astype(float), index=lives.index, columns=n),
              "predictive_lo": trigger_cycles(preds, "lo", np.arange(0, 11))}
    cfg = {"seed": 0, "cost_sim": {"ratio_grid": [1, 5], "n_folds": 4, "bootstrap": 50}}
    s = sensitivity(lives, tables, C, cfg, "ratio")
    assert s.loc[1, "best_alternative"] == "run_to_failure" and s.loc[1, "savings_vs_best_alternative"] <= 0
    assert s.loc[5, "savings_vs_best_alternative"] > 0
