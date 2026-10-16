"""FastAPI service: remaining life, confidence range, early-warning flag, top 3 drivers, recommended action.

Input is an engine's FULL history since install (cycle 1, 2, 3, ...), not just a recent window:
the early-warning index and the "distance from healthy start" features compare the engine with
its own first cycles. Models and settings come from models/ (written by `python -m src.train`).
"""
import json
import math
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.anomaly import persistent_alarm, prepare, zscore
from src.data import ROOT, load_config
from src.explain import top_drivers
from src.features import add_rolling_features
from src.field_note import make_note

try:  # local dev: read secrets (GROQ_API_KEY) from .env; Docker passes them with --env-file instead
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # never overrides variables already set in the environment
except ImportError:
    pass

CFG = load_config()
MODELS = ROOT / CFG["paths"]["models"]

app = FastAPI(title="Predictive maintenance API (NASA C-MAPSS stand-in data)", version="1.0")


@lru_cache(maxsize=1)
def artifacts() -> dict:
    """Load saved models once; FileNotFoundError if src.train has not been run."""
    meta = json.loads((MODELS / "meta.json").read_text())
    load = lambda name: lgb.Booster(model_file=str(MODELS / name))
    return {"meta": meta, "rul": load("rul.txt"), "lo": load("rul_lo.txt"), "hi": load("rul_hi.txt")}


class PredictRequest(BaseModel):
    # Short id only: it is echoed into the field note and the LLM prompt.
    engine_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9 ._-]*$")
    history: list[dict[str, float]] = Field(
        min_length=1, max_length=CFG["serving"]["max_history_rows"],
        description="One row per cycle from cycle 1, each with 'cycle' and every model sensor (e.g. 's2').",
    )


def validate_history(rows: list[dict[str, float]], sensors: list[str]) -> pd.DataFrame:
    """Reject input the models can't use: missing sensors, non-finite numbers, cycles not 1, 2, 3, ..."""
    df = pd.DataFrame(rows)
    missing = [c for c in ["cycle", *sensors] if c not in df.columns or df[c].isna().any()]
    if missing:
        raise HTTPException(422, f"every row needs 'cycle' and sensors {sensors}; missing or empty: {missing}")
    if not all(math.isfinite(v) for v in df[["cycle", *sensors]].to_numpy().ravel()):
        raise HTTPException(422, "all values must be finite numbers (no NaN or infinity)")
    if df["cycle"].tolist() != list(range(1, len(df) + 1)):
        raise HTTPException(422, "cycles must run 1, 2, 3, ... with no gaps: the model compares each engine "
                                 "with its own first cycles and uses the last N cycles for trends")
    return df[["cycle", *sensors]].astype(float).assign(cycle=lambda d: d["cycle"].astype(int), engine=0)


def recommend(lowest_lower_bound: float, alarm: bool, rules: dict) -> dict:
    """Action from rules validated earlier: Phase 5 cost-chosen threshold, Phase 2 early-warning alarm.

    lowest_lower_bound = lowest lower bound over the engine's history so far: once it reaches the
    threshold the replace decision sticks, exactly the policy the cost simulation evaluated.
    """
    t, lead = rules["replace_threshold"], rules["lead_cycles"]
    rule = {"replace_threshold": t, "lead_cycles": lead}
    if lowest_lower_bound <= t:
        return {"code": "replace", "text": f"Replace now: the lower bound has reached {lowest_lower_bound:.0f} cycles, "
                                           f"at or below the {t}-cycle threshold; allow about {lead} cycles to replace.",
                **rule}
    if alarm:
        return {"code": "plan", "text": "Plan: sensors have drifted from this engine's healthy start; "
                                        "order parts and schedule an inspection.", **rule}
    return {"code": "monitor", "text": "Monitor: no action needed now.", **rule}


def predict_engine(history: pd.DataFrame) -> dict:
    """Run every model on one validated engine history; answer for its latest cycle."""
    art = artifacts()
    m, an = art["meta"], art["meta"]["anomaly"]
    feats, _ = add_rolling_features(history, m["sensors"], m["window_sizes"], m["baseline_cycles"])
    X_all = feats[m["features"]]

    # remaining life + calibrated range (same arithmetic as src.rul.fit_predict), range for every cycle
    rul = float(art["rul"].predict(X_all.iloc[[-1]])[0])
    q = np.sort(np.column_stack([art["lo"].predict(X_all), art["hi"].predict(X_all)]), axis=1)
    lo_all = np.clip(q[:, 0] - m["conformal_margin"], 0.0, None)
    lo, hi = float(lo_all[-1]), float(q[-1, 1] + m["conformal_margin"])

    # early warning: z-score health index, alarm after `persistence` cycles above threshold
    health = zscore(prepare(history, m["sensors"], m["baseline_cycles"], an["smooth_window"]), an)
    above = pd.Series((health > an["threshold"]) & (history["cycle"] > m["baseline_cycles"]).to_numpy(), index=history.index)
    alarm = bool(persistent_alarm(above, history["engine"], an["persistence"]).iloc[-1])

    return {
        "cycles_observed": int(history["cycle"].iloc[-1]),
        "remaining_life": round(rul, 1),
        "range": {"low": round(lo, 1), "high": round(hi, 1), "coverage": m["interval_coverage"]},
        "early_warning": {"alarm": alarm, "health_index": round(float(health[-1]), 2), "threshold": round(an["threshold"], 2),
                          "monitoring_started": bool(history["cycle"].iloc[-1] > m["baseline_cycles"])},
        "top_drivers": top_drivers(art["rul"], feats.iloc[[-1]], CFG["sensor_names"]),
        "action": recommend(float(lo_all.min()), alarm, m["action"]),
        "health_index_history": [round(float(v), 3) for v in health],
        "model": {"trained": m["trained"], "dataset": m["dataset"]},
    }


@app.get("/health")
def health() -> dict:
    try:
        artifacts()
        return {"status": "ok", "models_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "models_loaded": False, "hint": "run `python -m src.train` to build models/"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        sensors = artifacts()["meta"]["sensors"]
    except FileNotFoundError:
        raise HTTPException(503, "models not built: run `python -m src.train`")
    return {"engine_id": req.engine_id, **predict_engine(validate_history(req.history, sensors))}


@app.post("/field-note")
def field_note(req: PredictRequest) -> dict:
    """Prediction plus a short note for a field engineer (LLM if GROQ_API_KEY is set, else template).

    Takes the history, not a prediction: the note can only use numbers this service computed itself.
    """
    pred = predict(req)
    return {**pred, "field_note": make_note(pred, CFG["field_note"])}
