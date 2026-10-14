"""Remaining-life models (mean, linear, LightGBM), conformal confidence ranges, and metrics."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from src.data import engine_folds

POINT_MODELS = ["mean", "linear", "lgbm"]


# ---------- metrics ----------
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def nasa_score(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-prediction NASA penalty; d = predicted - true.

    Late calls (d > 0: more life claimed than real) cost exp(d/10) - 1,
    early calls cost exp(-d/13) - 1, so being late hurts more.
    """
    d = np.asarray(p) - np.asarray(y)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------- conformal ----------
def conformal_margin(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: float) -> float:
    """Split-conformal (CQR) margin: widen [lo, hi] by this to reach `target` coverage.

    Score = how far the truth falls outside the range (negative if inside).
    The margin is the finite-sample-corrected `target` quantile of those scores.
    """
    scores = np.maximum(np.asarray(lo) - y, np.asarray(y) - hi)
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * target) / n), method="higher"))


# ---------- models ----------
def fit_lgbm(df: pd.DataFrame, features: list[str], cfg: dict) -> LGBMRegressor:
    """Main point model: LightGBM on rolling features, trained on capped RUL."""
    return LGBMRegressor(**cfg["rul"]["lgbm"], random_state=cfg["seed"]).fit(df[features], df["rul_capped"])


def _fit_quantiles(df: pd.DataFrame, features: list[str], cfg: dict) -> tuple[LGBMRegressor, LGBMRegressor]:
    """LightGBM models for the lower and upper quantile of the target range."""
    alpha = 1 - cfg["rul"]["interval_coverage"]
    return tuple(
        LGBMRegressor(**cfg["rul"]["lgbm"], objective="quantile", alpha=q, random_state=cfg["seed"])
        .fit(df[features], df["rul_capped"])
        for q in (alpha / 2, 1 - alpha / 2)
    )


def _predict_range(models: tuple, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = (m.predict(X) for m in models)
    return np.minimum(lo, hi), np.maximum(lo, hi)  # guard against crossing quantiles


def fit_range(fit_df: pd.DataFrame, features: list[str], cfg: dict) -> tuple[tuple, float]:
    """Cross-conformal confidence range: (final lower/upper quantile models, margin).

    Quantile models are fit inside engine folds of fit_df, so every fit engine gets an
    out-of-fold range; how often those ranges miss sets the margin. Final models use all fit engines.
    """
    r, cap = cfg["rul"], cfg["data"]["rul_cap"]
    y = fit_df["rul_capped"].to_numpy()
    oof_lo, oof_hi = np.empty(len(fit_df)), np.empty(len(fit_df))
    for in_fit, in_cal in engine_folds(fit_df, r["calibration_folds"]):
        models = _fit_quantiles(fit_df.iloc[in_fit], features, cfg)
        oof_lo[in_cal], oof_hi[in_cal] = _predict_range(models, fit_df.iloc[in_cal][features])
    # Calibrate where decisions happen (true RUL <= cap), the same region we evaluate.
    near = (fit_df["rul"] <= cap).to_numpy()
    margin = conformal_margin(y[near], oof_lo[near], oof_hi[near], r["interval_coverage"])
    return _fit_quantiles(fit_df, features, cfg), margin


def fit_predict(fit_df: pd.DataFrame, pred_df: pd.DataFrame, features: list[str], sensors: list[str], cfg: dict) -> pd.DataFrame:
    """Fit all models on fit_df, predict pred_df.

    Returns columns: mean, linear, lgbm, lo_raw, hi_raw, lo, hi (range: see fit_range).
    """
    y = fit_df["rul_capped"]
    out = pd.DataFrame(index=pred_df.index)
    out["mean"] = y.mean()
    out["linear"] = LinearRegression().fit(fit_df[sensors], y).predict(pred_df[sensors])
    out["lgbm"] = fit_lgbm(fit_df, features, cfg).predict(pred_df[features])

    models, margin = fit_range(fit_df, features, cfg)
    lo_pred, hi_pred = _predict_range(models, pred_df[features])
    out["lo_raw"], out["hi_raw"] = lo_pred, hi_pred
    out["lo"] = np.clip(lo_pred - margin, 0, None)  # RUL can't be negative
    out["hi"] = hi_pred + margin
    out.attrs["margin"] = margin
    return out


def cross_validate(train: pd.DataFrame, features: list[str], sensors: list[str], cfg: dict) -> pd.DataFrame:
    """Engine-grouped CV: every engine predicted by models that never saw it."""
    parts = []
    for fit_idx, val_idx in engine_folds(train, cfg["rul"]["n_folds"]):
        p = fit_predict(train.iloc[fit_idx], train.iloc[val_idx], features, sensors, cfg)
        parts.append(p.assign(margin=p.attrs["margin"]))
    preds = pd.concat(parts).loc[train.index]
    return pd.concat([train[["engine", "cycle", "rul", "rul_capped"]], preds], axis=1)


def summarize(preds: pd.DataFrame, y_col: str = "rul") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(point-model table, interval table) against true RUL in `y_col`."""
    y = preds[y_col].to_numpy()
    point = pd.DataFrame({
        m: {"rmse": rmse(y, preds[m]), "nasa_mean": nasa_score(y, preds[m]).mean(),
            "nasa_sum": nasa_score(y, preds[m]).sum(), "n": len(y)}
        for m in POINT_MODELS
    }).T
    interval = pd.DataFrame({
        k: {"coverage": coverage(y, preds[lo], preds[hi]), "mean_width": (preds[hi] - preds[lo]).mean()}
        for k, (lo, hi) in {"quantile_raw": ("lo_raw", "hi_raw"), "conformal": ("lo", "hi")}.items()
    }).T
    return point, interval
