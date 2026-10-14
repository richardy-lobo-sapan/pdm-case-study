"""Maintenance strategy simulation: run-to-failure vs fixed schedule vs predictive.

All costs are ASSUMPTIONS from configs/costs.yaml. Each engine fails at its last cycle (life L).
A strategy picks a replacement cycle; replacing at or after L means the engine failed first.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import ROOT, engine_folds

STRATEGIES = ["run_to_failure", "fixed", "predictive_lo", "predictive_point"]
LABELS = {
    "run_to_failure": "Run to failure",
    "fixed": "Fixed schedule",
    "predictive_lo": "Predictive (lower bound)",
    "predictive_point": "Predictive (point estimate)",
}


def load_costs(path: Path = ROOT / "configs" / "costs.yaml") -> dict:
    """Read ASSUMED costs."""
    return yaml.safe_load(path.read_text())


def event_costs(c: dict) -> tuple[float, float]:
    """(cost of one unplanned failure, cost of one planned replacement), downtime included."""
    failure = c["unplanned_failure_cost"] + c["downtime_cost_per_cycle"] * c["unplanned_downtime_cycles"]
    planned = c["planned_replacement_cost"] + c["downtime_cost_per_cycle"] * c["planned_downtime_cycles"]
    return failure, planned


def engine_costs(replace_at: np.ndarray, life: np.ndarray, c: dict) -> np.ndarray:
    """Cost per engine: failure if replace_at >= life, else planned + value of life thrown away."""
    failure, planned = event_costs(c)
    with np.errstate(invalid="ignore"):  # inf - inf in the unused branch
        wasted = np.where(replace_at >= life, 0.0, life - replace_at)
    return np.where(replace_at >= life, failure, planned + c["wasted_life_cost_per_cycle"] * wasted)


def trigger_cycles(preds: pd.DataFrame, signal: str, thresholds: np.ndarray) -> pd.DataFrame:
    """First cycle each engine's `signal` (predicted RUL) is <= each threshold; inf if never.

    Table: rows = engines, columns = thresholds.
    """
    out = {}
    for eng, g in preds.groupby("engine"):
        running_min = np.minimum.accumulate(g[signal].to_numpy())
        hit = running_min[None, :] <= thresholds[:, None]
        out[eng] = np.where(hit.any(axis=1), g["cycle"].to_numpy()[hit.argmax(axis=1)], np.inf)
    return pd.DataFrame(out, index=thresholds).T.rename_axis("engine")


def schedules(preds: pd.DataFrame, cap: int) -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    """Engine lives, and per strategy a table (engines x setting) of the planned replacement cycle.

    Settings: fixed = replace every N cycles; predictive = RUL threshold T (0..cap).
    Lead time is added later, so these tables can be reused across lead-time scenarios.
    """
    lives = preds.groupby("engine")["cycle"].max()
    n_grid = np.arange(1, lives.max() + 1)
    t_grid = np.arange(0, cap + 1)
    tables = {
        "run_to_failure": pd.DataFrame(np.inf, index=lives.index, columns=[0]),
        "fixed": pd.DataFrame(np.tile(n_grid, (len(lives), 1)).astype(float), index=lives.index, columns=n_grid),
        "predictive_lo": trigger_cycles(preds, "lo", t_grid),
        "predictive_point": trigger_cycles(preds, "lgbm", t_grid),
    }
    return lives, tables


def _replace_at(strategy: str, table: pd.DataFrame, lead: int) -> pd.DataFrame:
    return table + lead if strategy.startswith("predictive") else table


def evaluate(lives: pd.Series, tables: dict, c: dict, lead: int, n_folds: int) -> pd.DataFrame:
    """Per engine: each strategy's setting, replacement cycle and cost.

    The setting (N or T) for each engine is the one with the lowest mean cost on the
    *other* folds' engines, so no engine's outcome influences its own setting.
    """
    folds = list(engine_folds(lives.reset_index()[["engine"]], n_folds))
    life = lives.to_numpy()[:, None]
    rows = pd.DataFrame({"life": lives})
    for s, table in tables.items():
        replace_at = _replace_at(s, table, lead)
        cost = pd.DataFrame(engine_costs(replace_at.to_numpy(), life, c), index=table.index, columns=table.columns)
        chosen = np.empty(len(lives), dtype=object)
        for fit_idx, val_idx in folds:
            chosen[val_idx] = cost.iloc[fit_idx].mean().idxmin()
        rows[f"{s}_setting"] = chosen
        rows[f"{s}_replace_at"] = [replace_at.at[e, p] for e, p in zip(lives.index, chosen)]
        rows[f"{s}_cost"] = [cost.at[e, p] for e, p in zip(lives.index, chosen)]
        rows[f"{s}_failed"] = rows[f"{s}_replace_at"] >= rows["life"]
        rows[f"{s}_wasted"] = np.where(rows[f"{s}_failed"], 0, rows["life"] - rows[f"{s}_replace_at"])
    return rows


def best_settings(lives: pd.Series, tables: dict, c: dict, lead: int) -> dict:
    """Setting with the lowest mean cost on all engines (the one we'd deploy)."""
    life = lives.to_numpy()[:, None]
    return {
        s: int(pd.Series(engine_costs(_replace_at(s, t, lead).to_numpy(), life, c).mean(axis=0), index=t.columns).idxmin())
        for s, t in tables.items()
    }


def cost_per_operating_cycle(rows: pd.DataFrame, s: str, c: dict) -> float:
    """Event costs (no wasted-life charge) / cycles actually run: the classic renewal-rate view.

    Cross-check that needs no price for wasted life: replacing early simply means
    paying for the next replacement sooner.
    """
    failure, planned = event_costs(c)
    failed = rows[f"{s}_failed"]
    events = np.where(failed, failure, planned).sum()
    cycles_run = np.where(failed, rows["life"], rows[f"{s}_replace_at"]).sum()
    return float(events / cycles_run)


def summarize(rows: pd.DataFrame, c: dict) -> pd.DataFrame:
    """One row per strategy: cost per engine, failures, planned replacements, wasted life, cost per cycle."""
    out = {}
    for s in STRATEGIES:
        planned = ~rows[f"{s}_failed"]
        settings = rows[f"{s}_setting"].astype(float)
        out[LABELS[s]] = {
            "mean_cost_per_engine": rows[f"{s}_cost"].mean(),
            "cost_per_operating_cycle": cost_per_operating_cycle(rows, s, c),
            "failures": int(rows[f"{s}_failed"].sum()),
            "planned_replacements": int(planned.sum()),
            "mean_wasted_cycles_when_planned": rows.loc[planned, f"{s}_wasted"].mean() if planned.any() else 0.0,
            "setting_range_across_folds": f"{settings.min():.0f}-{settings.max():.0f}",
        }
    return pd.DataFrame(out).T


def savings(rows: pd.DataFrame, c: dict, n_boot: int, seed: int, predictive: str = "predictive_lo") -> dict:
    """Savings of `predictive` vs the cheapest non-predictive strategy (and vs run-to-failure),
    with a bootstrap 95% range from resampling engines, plus the cost-per-cycle cross-check."""
    means = {s: rows[f"{s}_cost"].mean() for s in ["run_to_failure", "fixed", predictive]}
    best_alt = min(["run_to_failure", "fixed"], key=means.get)
    pct = lambda base, new: 1 - new / base
    rng = np.random.default_rng(seed)
    alt, pred = rows[f"{best_alt}_cost"].to_numpy(), rows[f"{predictive}_cost"].to_numpy()
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    boot = 1 - pred[idx].mean(axis=1) / alt[idx].mean(axis=1)
    return {
        "best_alternative": best_alt,
        "savings_vs_best_alternative": pct(means[best_alt], means[predictive]),
        "savings_vs_best_alternative_95range": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "savings_vs_run_to_failure": pct(means["run_to_failure"], means[predictive]),
        "savings_per_operating_cycle_vs_best_alternative": pct(
            cost_per_operating_cycle(rows, best_alt, c), cost_per_operating_cycle(rows, predictive, c)),
        "mean_cost_per_engine": means,
    }


def sensitivity(lives: pd.Series, tables: dict, c: dict, cfg: dict, param: str) -> pd.DataFrame:
    """Re-choose settings and recompute savings while varying one assumption.

    param = "ratio": unplanned failure cost / planned replacement cost (downtime fixed).
    param = "lead":  maintenance lead time in cycles.
    param = "waste": value of one wasted remaining-life cycle.
    """
    sc = cfg["cost_sim"]
    rows = []
    for v in sc[f"{param}_grid"]:
        cc, lead = dict(c), c["maintenance_lead_cycles"]
        if param == "ratio":
            cc["unplanned_failure_cost"] = v * c["planned_replacement_cost"]
        elif param == "waste":
            cc["wasted_life_cost_per_cycle"] = v
        else:
            lead = int(v)
        r = evaluate(lives, tables, cc, lead, sc["n_folds"])
        s = savings(r, cc, sc["bootstrap"], cfg["seed"])
        rows.append({param: v, "best_alternative": s["best_alternative"],
                     "savings_vs_best_alternative": s["savings_vs_best_alternative"],
                     "low95": s["savings_vs_best_alternative_95range"][0],
                     "high95": s["savings_vs_best_alternative_95range"][1],
                     "savings_per_operating_cycle": s["savings_per_operating_cycle_vs_best_alternative"],
                     "predictive_failures": int(r["predictive_lo_failed"].sum())})
    return pd.DataFrame(rows).set_index(param)
