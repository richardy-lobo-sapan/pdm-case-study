"""SHAP explanations for the LightGBM remaining-life model, grouped by sensor, in plain words.

Uses LightGBM's built-in exact TreeSHAP (pred_contrib=True): same numbers as the `shap`
package, without its ~40 MB of dependencies.
"""
from __future__ import annotations  # hints below are not evaluated: the API image has no scikit-learn

import lightgbm as lgb
import pandas as pd

Model = "lgb.Booster | lgb.LGBMRegressor"  # training-time model or one loaded from a saved text file


def feature_names(model: Model) -> list[str]:
    """Feature columns the model expects, for either model type."""
    return model.feature_name() if isinstance(model, lgb.Booster) else model.feature_name_


def contributions(model: Model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature push on the prediction (cycles) plus a 'base' column (fleet average).

    For every row: base + sum of pushes = model prediction, exactly.
    """
    raw = model.predict(X, pred_contrib=True)
    return pd.DataFrame(raw, index=X.index, columns=[*X.columns, "base"])


def sensor_of(feature: str) -> str:
    """'s11_slope10' -> 's11'; 'cycle' -> 'cycle'."""
    return feature.split("_")[0]


def by_sensor(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum feature pushes per sensor: one column per sensor (and 'cycle'), base dropped."""
    return contrib.drop(columns="base").T.groupby(sensor_of).sum().T


def top_drivers(model: Model, row: pd.DataFrame, sensor_names: dict, k: int = 3) -> list[dict]:
    """Top-k sensors pushing one prediction, largest push first, each with a plain-English sentence.

    row: a one-row DataFrame of model features (must include '<sensor>_dev' columns).
    effect_cycles < 0 means the sensor shortens the estimate, compared with the fleet-average
    engine (SHAP's starting point) — not compared with this engine's own past.
    """
    pushes = by_sensor(contributions(model, row[feature_names(model)])).iloc[0]
    drivers = []
    for sensor in pushes.abs().sort_values(ascending=False).index[:k]:
        effect = round(float(pushes[sensor]), 1)
        name, unit = sensor_names[sensor]["name"], sensor_names[sensor]["unit"]
        push = f"{'shortens' if effect < 0 else 'extends'} the estimate by {abs(effect)} cycles compared with the average engine"
        if sensor == "cycle":
            reading = round(float(row["cycle"].iloc[0]))
            text = f"Engine age ({reading} cycles run) {push}."
        else:
            reading = round(float(row[f"{sensor}_dev"].iloc[0]), 2)
            side = "above" if reading >= 0 else "below"
            text = (f"{name[0].upper() + name[1:]} ({sensor}) is {abs(reading)} {unit} {side} this engine's "
                    f"healthy start; this {push}.").replace("  ", " ")
        drivers.append({"sensor": sensor, "name": name, "effect_cycles": effect, "reading": reading, "text": text})
    return drivers
