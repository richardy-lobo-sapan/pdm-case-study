"""Early-warning scores: z-score health index (baseline), PCA reconstruction error, Isolation Forest.

Both score how far an engine has drifted from its own healthy start. An alarm fires when
the score stays above a threshold learned from healthy training data.
"""
import numpy as np
import pandas as pd

from src.data import engine_folds

METHODS = ["zscore", "pca", "iforest"]


def prepare(df: pd.DataFrame, sensors: list[str], baseline_cycles: int, smooth_window: int) -> pd.DataFrame:
    """Deviation from each engine's own healthy mean, then a trailing (past-only) rolling mean.

    Subtracting the engine's own early mean removes engine-to-engine offsets.
    Only that engine's past is used, so nothing leaks from the future.
    """
    base = df[df["cycle"] <= baseline_cycles].groupby("engine")[sensors].mean()
    dev = df[sensors] - base.loc[df["engine"]].to_numpy()
    return dev.groupby(df["engine"]).transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())


def healthy_mask(df: pd.DataFrame, baseline_cycles: int, healthy_rul_min: int) -> pd.Series:
    """Rows after the baseline window that are still far from failure."""
    return (df["cycle"] > baseline_cycles) & (df["rul"] > healthy_rul_min)


def zscore_params(X_fit: pd.DataFrame, rul_fit: pd.Series, healthy: pd.Series,
                  threshold_quantile: float | None = None) -> dict:
    """Fitted z-score health index as plain numbers (saved to models/meta.json for the API).

    scale: noise level of each sensor on healthy rows.
    direction: +1 if the sensor rises toward failure, -1 if it falls, so "higher = worse".
    threshold: optional alarm threshold = quantile of the index on healthy rows.
    """
    p = {"scale": X_fit[healthy].std().to_dict(),
         "direction": (-np.sign(X_fit.corrwith(rul_fit, method="spearman"))).to_dict()}
    if threshold_quantile is not None:
        p["threshold"] = float(np.quantile(zscore(X_fit[healthy], p), threshold_quantile))
    return p


def zscore(X: pd.DataFrame, p: dict) -> np.ndarray:
    """Health index: average per-sensor distance from healthy start, in noise units, higher = worse."""
    return (X / pd.Series(p["scale"]) * pd.Series(p["direction"])).mean(axis=1).to_numpy()


def fit_score(X_fit: pd.DataFrame, rul_fit: pd.Series, healthy: pd.Series, pca_variance: float, seed: int) -> dict:
    """Learn z-score parameters, PCA and Isolation Forest from training rows; return scoring functions."""
    from sklearn.decomposition import PCA  # imported here so the API image needs no scikit-learn
    from sklearn.ensemble import IsolationForest

    p = zscore_params(X_fit, rul_fit, healthy)
    scale = pd.Series(p["scale"])
    Z_healthy = (X_fit[healthy] / scale).to_numpy()
    pca = PCA(n_components=pca_variance, svd_solver="full").fit(Z_healthy)
    forest = IsolationForest(random_state=seed).fit(Z_healthy)

    def pca_error(X: pd.DataFrame) -> np.ndarray:
        Z = (X / scale).to_numpy()
        return ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).sum(axis=1)

    def iforest(X: pd.DataFrame) -> np.ndarray:
        return -forest.score_samples((X / scale).to_numpy())  # flip so higher = more anomalous

    return {"zscore": lambda X: zscore(X, p), "pca": pca_error, "iforest": iforest, "n_components": pca.n_components_}


def persistent_alarm(above: pd.Series, engine: pd.Series, persistence: int) -> pd.Series:
    """True once the score has been above threshold for `persistence` cycles in a row (per engine)."""
    run = above.astype(float).groupby(engine).transform(lambda s: s.rolling(persistence).min())
    return run.eq(1)


def cross_validate(train: pd.DataFrame, sensors: list[str], cfg: dict) -> pd.DataFrame:
    """Engine-grouped CV: every engine is scored by models fit on other engines only.

    Returns engine, cycle, rul plus <method>_score, <method>_threshold and <method>_alarm columns.
    """
    a = cfg["anomaly"]
    X = prepare(train, sensors, a["baseline_cycles"], a["smooth_window"])
    healthy = healthy_mask(train, a["baseline_cycles"], a["healthy_rul_min"])
    monitored = (train["cycle"] > a["baseline_cycles"]).to_numpy()  # no alarms during the baseline window
    score = {m: np.full(len(train), np.nan) for m in METHODS}
    thresh = {m: np.full(len(train), np.nan) for m in METHODS}
    above = {m: np.zeros(len(train), dtype=bool) for m in METHODS}

    for fit_idx, val_idx in engine_folds(train, a["n_folds"]):
        X_fit, h_fit = X.iloc[fit_idx], healthy.iloc[fit_idx]
        scorers = fit_score(X_fit, train["rul"].iloc[fit_idx], h_fit, a["pca_variance"], cfg["seed"])
        for m in METHODS:
            thresh[m][val_idx] = np.quantile(scorers[m](X_fit[h_fit]), a["threshold_quantile"])
            score[m][val_idx] = scorers[m](X.iloc[val_idx])
            above[m][val_idx] = score[m][val_idx] > thresh[m][val_idx]

    out = train[["engine", "cycle", "rul"]].copy()
    for m in METHODS:
        out[f"{m}_score"] = score[m]
        out[f"{m}_threshold"] = thresh[m]
        out[f"{m}_alarm"] = persistent_alarm(pd.Series(above[m] & monitored, index=out.index), out["engine"], a["persistence"])
    return out


def engine_summary(scored: pd.DataFrame, method: str, baseline_cycles: int, healthy_rul_min: int) -> pd.DataFrame:
    """Per engine: false alarm (alarm while clearly healthy) and lead time (first alarm in final window).

    lead_time = cycles before failure of the first alarm with RUL <= healthy_rul_min (NaN = missed).
    Capped at healthy_rul_min by definition.
    """
    alarm = scored[f"{method}_alarm"]
    healthy = healthy_mask(scored, baseline_cycles, healthy_rul_min)
    g = scored.assign(
        false_alarm=alarm & healthy,
        warn_rul=scored["rul"].where(alarm & (scored["rul"] <= healthy_rul_min)),
        healthy_row=healthy,
    ).groupby("engine")
    return pd.DataFrame({
        "has_healthy_period": g["healthy_row"].any(),
        "false_alarm": g["false_alarm"].any(),
        "lead_time": g["warn_rul"].max(),
    })


def summarize(scored: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per method: lead-time stats and false alarm rates on held-out engines."""
    a = cfg["anomaly"]
    healthy = healthy_mask(scored, a["baseline_cycles"], a["healthy_rul_min"])
    rows = {}
    for m in METHODS:
        e = engine_summary(scored, m, a["baseline_cycles"], a["healthy_rul_min"])
        rows[m] = {
            "engines": len(e),
            "warned": int(e["lead_time"].notna().sum()),
            "missed": int(e["lead_time"].isna().sum()),
            "median_lead": e["lead_time"].median(),
            "p10_lead": e["lead_time"].quantile(0.10),  # 90% of warned engines get at least this
            "engines_with_healthy_period": int(e["has_healthy_period"].sum()),
            "engine_false_alarm_rate": e.loc[e["has_healthy_period"], "false_alarm"].mean(),
            "row_false_alarm_rate": scored.loc[healthy, f"{m}_alarm"].mean(),
        }
    return pd.DataFrame(rows).T
