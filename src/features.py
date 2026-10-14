"""Backward-looking rolling-window features for remaining-life models."""
import pandas as pd


def _grouped_rolling_mean(values: pd.DataFrame, engine: pd.Series, window: int) -> pd.DataFrame:
    """Trailing mean per engine over the last `window` rows (fewer at the start)."""
    out = values.groupby(engine).rolling(window, min_periods=1).mean()
    return out.reset_index(level=0, drop=True).loc[values.index]


def add_rolling_features(
    df: pd.DataFrame, sensors: list[str], windows: list[int], baseline_cycles: int
) -> tuple[pd.DataFrame, list[str]]:
    """Add per-sensor rolling mean/std/slope, deviation from own baseline, and cycle.

    Every feature at cycle t uses only cycles <= t of the same engine, matching what
    is known in real use. Returns (df with features, feature column names).
    """
    engine, t = df["engine"], df["cycle"].astype(float)
    g = df.groupby("engine")[sensors]
    feats = {"cycle": t}

    # Baseline = mean of cycles 1..min(t, baseline_cycles): never peeks past cycle t.
    expanding = g.expanding().mean().reset_index(level=0, drop=True).loc[df.index]
    baseline = expanding.where(df["cycle"] <= baseline_cycles).groupby(engine).ffill()
    for s in sensors:
        feats[f"{s}_dev"] = df[s] - baseline[s]

    tx = df[sensors].mul(t, axis=0)
    for w in windows:
        m_x = _grouped_rolling_mean(df[sensors], engine, w)
        m_t = _grouped_rolling_mean(t.to_frame(), engine, w)["cycle"]
        m_tt = _grouped_rolling_mean((t * t).to_frame(), engine, w)["cycle"]
        m_tx = _grouped_rolling_mean(tx, engine, w)
        var_t = (m_tt - m_t**2).where(lambda v: v > 0)  # first row of an engine: no slope yet
        std = g.rolling(w, min_periods=2).std().reset_index(level=0, drop=True).loc[df.index]
        for s in sensors:
            feats[f"{s}_mean{w}"] = m_x[s]
            feats[f"{s}_std{w}"] = std[s]
            feats[f"{s}_slope{w}"] = (m_tx[s] - m_x[s] * m_t) / var_t  # least-squares slope per cycle

    feat_df = pd.DataFrame(feats, index=df.index)
    return pd.concat([df, feat_df.drop(columns="cycle")], axis=1), list(feat_df.columns)
