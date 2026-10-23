"""Full pipeline: data -> early warning -> remaining life -> explanations -> cost simulation -> saved models.

Writes reports/results.json (single source of truth for deck numbers),
data/processed/rul_oof.csv (cross-validated RUL predictions, reused by notebooks),
models/ (LightGBM text files + meta.json for the API) and one MLflow run (local sqlite file).
Run: python -m src.train
"""
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import anomaly, cost_sim, explain, rul
from src.data import ROOT, SENSORS, add_rul, add_test_rul, load_config, load_fd, near_constant_columns
from src.features import add_rolling_features


def _records(df: pd.DataFrame) -> dict:
    return json.loads(df.to_json(orient="index"))


def save_models(out: Path, point, range_models: tuple, meta: dict) -> None:
    """LightGBM models as text files (readable, no pickle) plus everything the API needs in meta.json."""
    out.mkdir(parents=True, exist_ok=True)
    point.booster_.save_model(str(out / "rul.txt"))
    range_models[0].booster_.save_model(str(out / "rul_lo.txt"))
    range_models[1].booster_.save_model(str(out / "rul_hi.txt"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


def log_mlflow(cfg: dict, costs: dict, results: dict, files: list[Path]) -> None:
    """One MLflow run: settings, headline metrics, results.json and the models (local sqlite store)."""
    import mlflow  # dev-only dependency; not needed by the API image

    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])
    r, c = results["rul"], results["cost"]
    with mlflow.start_run():
        mlflow.log_params({"rul_cap": cfg["data"]["rul_cap"], "window_sizes": cfg["features"]["window_sizes"],
                           **{f"lgbm_{k}": v for k, v in cfg["rul"]["lgbm"].items()},
                           **{f"anomaly_{k}": v for k, v in cfg["anomaly"].items()},
                           **{f"cost_{k}": v for k, v in costs.items()}})
        mlflow.log_metrics({
            "cv_rmse": r["cv_rows_rul_le_cap"]["point"]["lgbm"]["rmse"],
            "cv_coverage": r["cv_rows_rul_le_cap"]["interval"]["conformal"]["coverage"],
            "test_rmse": r["test_last_cycle"]["point"]["lgbm"]["rmse"],
            "test_rmse_ridge": r["test_last_cycle"]["point"]["ridge"]["rmse"],
            "test_rmse_seed_std": r["test_lgbm_rmse_by_seed"]["std"],
            "test_nasa_sum": r["test_last_cycle"]["point"]["lgbm"]["nasa_sum"],
            "test_coverage": r["test_last_cycle"]["interval"]["conformal"]["coverage"],
            "warning_median_lead": results["early_warning_cv"]["zscore"]["median_lead"],
            "warning_engine_false_alarm_rate": results["early_warning_cv"]["zscore"]["engine_false_alarm_rate"],
            "savings_vs_fixed": c["savings_lower_bound_policy"]["savings_vs_best_alternative"],
            "savings_per_cycle_vs_fixed": c["savings_lower_bound_policy"]["savings_per_operating_cycle_vs_best_alternative"],
        })
        for f in files:
            mlflow.log_artifact(str(f))


def main() -> None:
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)
    cfg, costs = load_config(), cost_sim.load_costs()
    cap = cfg["data"]["rul_cap"]

    train, test, true_rul = load_fd(cfg["data"]["subset"])
    train, test = add_rul(train, cap), add_test_rul(test, true_rul, cap)
    dropped = near_constant_columns(train, cfg["data"]["near_constant_std"])
    sensors = [s for s in SENSORS if s not in dropped]
    feat_args = (sensors, cfg["features"]["window_sizes"], cfg["anomaly"]["baseline_cycles"])
    train, features = add_rolling_features(train, *feat_args)
    test, _ = add_rolling_features(test, *feat_args)
    log(f"data ready: {len(features)} features")

    anomaly_summary = anomaly.summarize(anomaly.cross_validate(train, sensors, cfg), cfg)
    log("early warning done")

    oof = rul.cross_validate(train, features, sensors, cfg)
    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "rul_oof.csv", index=False)
    cv_point, cv_interval = rul.summarize(oof[oof["rul"] <= cap])
    cv_bands = rul.coverage_by_band(oof[oof["rul"] <= cap], cfg["rul"]["coverage_bands"])
    last = test.groupby("engine").tail(1)
    test_pred = pd.concat([last[["engine", "rul"]], rul.fit_predict(train, last, features, sensors, cfg)], axis=1)
    test_point, test_interval = rul.summarize(test_pred)
    # same data, other seeds: how much of the test RMSE is luck of the draw (bagging/column sampling)
    seed_rmse = [rul.rmse(last["rul"], rul.fit_lgbm(train, features, {**cfg, "seed": cfg["seed"] + i}).predict(last[features]))
                 for i in range(cfg["rul"]["seed_runs"])]
    log("remaining life done")

    model = rul.fit_lgbm(train, features, cfg)
    imp = explain.by_sensor(explain.contributions(model, test[features])).abs().mean()
    share = (imp / imp.sum()).sort_values(ascending=False)
    log("explanations done")

    lives, tables = cost_sim.schedules(oof, cap)
    lead, sc = costs["maintenance_lead_cycles"], cfg["cost_sim"]
    rows = cost_sim.evaluate(lives, tables, costs, lead, sc["n_folds"])
    by_ratio = cost_sim.sensitivity(lives, tables, costs, cfg, "ratio")
    by_lead = cost_sim.sensitivity(lives, tables, costs, cfg, "lead")
    by_waste = cost_sim.sensitivity(lives, tables, costs, cfg, "waste")
    log("cost simulation done")

    results = {
        "meta": {
            "generated": date.today().isoformat(),
            "dataset": f"NASA C-MAPSS {cfg['data']['subset']} (simulation; stand-in for rotating equipment, not real field data)",
            "costs_are_assumptions": True,
        },
        "data": {"train_engines": int(train.engine.nunique()), "test_engines": int(test.engine.nunique()),
                 "dropped_columns": dropped, "sensors_used": sensors, "rul_cap": cap},
        "early_warning_cv": _records(anomaly_summary),
        "rul": {
            "cv_rows_rul_le_cap": {"point": _records(cv_point), "interval": _records(cv_interval),
                                   "interval_by_true_rul": _records(cv_bands)},
            "test_lgbm_rmse_by_seed": {"seeds": [cfg["seed"] + i for i in range(len(seed_rmse))], "rmse": seed_rmse,
                                       "mean": float(np.mean(seed_rmse)), "std": float(np.std(seed_rmse))},
            "test_last_cycle": {"point": _records(test_point), "interval": _records(test_interval),
                                "engines_failing_before_lower_bound": int((test_pred.rul < test_pred.lo).sum()),
                                "engines_outside_range": int(((test_pred.rul < test_pred.lo) | (test_pred.rul > test_pred.hi)).sum())},
        },
        "explanations": {"share_of_total_push_by_sensor": share.round(4).to_dict()},
        "cost": {
            "assumed_costs": costs,
            "event_costs": dict(zip(["failure", "planned"], cost_sim.event_costs(costs))),
            "strategies_held_out": _records(cost_sim.summarize(rows, costs)),
            "chosen_settings_all_engines": cost_sim.best_settings(lives, tables, costs, lead),
            "savings_lower_bound_policy": cost_sim.savings(rows, costs, sc["bootstrap"], cfg["seed"]),
            "savings_point_policy": cost_sim.savings(rows, costs, sc["bootstrap"], cfg["seed"], "predictive_point"),
            "sensitivity_failure_to_planned_ratio": _records(by_ratio),
            "sensitivity_lead_time": _records(by_lead),
            "sensitivity_wasted_life_value": _records(by_waste),
        },
    }
    path = ROOT / "reports" / "results.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    log(f"wrote {path.relative_to(ROOT)}")

    # Deployment models: same data and seed as the test evaluation above, so identical predictions.
    range_models, margin = rul.fit_range(train, features, cfg)
    X_last = last[features]
    q_lo, q_hi = (m.predict(X_last) for m in range_models)
    assert (np.allclose(model.predict(X_last), test_pred["lgbm"])
            and np.allclose(np.clip(np.minimum(q_lo, q_hi) - margin, 0, None), test_pred["lo"])
            and np.allclose(np.maximum(q_lo, q_hi) + margin, test_pred["hi"])), "deployed models differ from the evaluated ones"
    a = cfg["anomaly"]
    X_dev = anomaly.prepare(train, sensors, a["baseline_cycles"], a["smooth_window"])
    healthy = anomaly.healthy_mask(train, a["baseline_cycles"], a["healthy_rul_min"])
    check = test_pred.iloc[0]
    meta = {
        "trained": date.today().isoformat(), "dataset": results["meta"]["dataset"],
        "features": features, "sensors": sensors, "window_sizes": cfg["features"]["window_sizes"],
        # readings outside what the model was trained on get flagged by the API (sensor fault or a different machine)
        "sensor_ranges": {s: [float(train[s].min()), float(train[s].max())] for s in sensors},
        "baseline_cycles": a["baseline_cycles"], "rul_cap": cap,
        "interval_coverage": cfg["rul"]["interval_coverage"], "conformal_margin": margin,
        "anomaly": {"smooth_window": a["smooth_window"], "persistence": a["persistence"],
                    **anomaly.zscore_params(X_dev, train["rul"], healthy, a["threshold_quantile"])},
        "action": {"replace_threshold": results["cost"]["chosen_settings_all_engines"]["predictive_lo"],
                   "lead_cycles": lead},
        # the API must reproduce this for test engine 1's full history (tests/test_api.py)
        "parity_check": {"engine": int(check["engine"]), "rul": float(check["lgbm"]),
                         "lo": float(check["lo"]), "hi": float(check["hi"])},
    }
    models_dir = ROOT / cfg["paths"]["models"]
    save_models(models_dir, model, range_models, meta)
    log(f"saved models to {models_dir.relative_to(ROOT)}")

    log_mlflow(cfg, costs, results, [path, *sorted(models_dir.iterdir())])
    log("logged MLflow run")


if __name__ == "__main__":
    main()
