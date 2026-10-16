"""Streamlit dashboard: which engines need attention, and why. Every number comes from the FastAPI service.

Run the API first (uvicorn api.main:app, or the Docker image), then: streamlit run dashboard/app.py
Fleet = the 100 NASA C-MAPSS FD001 test engines (simulation data, used as a stand-in for rotating equipment).
"""
import html
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run` doesn't add the repo root

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from src.data import SENSORS, load_config, load_fd

CFG = load_config()
API_URL = os.environ.get("API_URL", CFG["serving"]["api_url"])
# Status colours are reserved for the three actions and always shown with an icon + label (never colour alone).
STATUS = {
    "replace": {"label": "Replace now", "icon": "⛔", "color": "#d03b3b", "ink": "#ffffff"},
    "plan": {"label": "Plan", "icon": "⚠️", "color": "#fab219", "ink": "#0b0b0b"},
    "monitor": {"label": "Monitor", "icon": "✅", "color": "#0ca30c", "ink": "#ffffff"},
}
SHORTENS, EXTENDS, NEUTRAL = "#eb6834", "#2a78d6", "#8a8984"
STATUS_SCALE = alt.Scale(domain=[s["label"] for s in STATUS.values()], range=[s["color"] for s in STATUS.values()])

st.set_page_config(page_title="Fleet health", page_icon="🛠️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1320px;}
  h1 {font-size: 2.0rem !important; margin-bottom: 0 !important;}
  .sub {opacity: .72; margin: .2rem 0 1.1rem 0;}
  .badge {display: inline-block; font-size: .72rem; padding: 2px 9px; margin-left: 6px; border-radius: 999px;
          border: 1px solid rgba(128,128,128,.45); opacity: .85; vertical-align: middle;}
  .kpi {border: 1px solid rgba(128,128,128,.22); border-left: 6px solid var(--c); border-radius: 12px;
        padding: 14px 18px; height: 100%;}
  .kpi .label {font-size: .82rem; opacity: .72; text-transform: uppercase; letter-spacing: .04em;}
  .kpi .value {font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin-top: 2px;}
  .kpi .hint {font-size: .8rem; opacity: .68;}
  .banner {border-radius: 12px; padding: 14px 18px; background: var(--c); color: var(--ink); font-size: 1.02rem;}
  .banner b {font-size: 1.12rem;}
  .note {border-left: 4px solid #2a78d6; background: rgba(42,120,214,.09); padding: 14px 18px;
         border-radius: 8px; line-height: 1.55;}
  .driver {padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,.15); line-height: 1.45;}
</style>
""", unsafe_allow_html=True)


def kpi(col, label: str, value: str, hint: str, color: str) -> None:
    col.markdown(f"<div class='kpi' style='--c:{color}'><div class='label'>{label}</div>"
                 f"<div class='value'>{value}</div><div class='hint'>{hint}</div></div>", unsafe_allow_html=True)


@st.cache_resource
def client() -> httpx.Client:
    """One reused connection: creating an httpx client costs ~1.8 s on this machine (SSL setup)."""
    return httpx.Client(base_url=API_URL, timeout=30)


@st.cache_data(show_spinner=False)
def fleet_histories() -> tuple[dict[int, list[dict]], pd.Series]:
    """Each test engine's full history (as the API expects) and its true remaining life (known only in this demo)."""
    _, test, true_rul = load_fd(CFG["data"]["subset"])
    cols = ["cycle", *SENSORS]
    return {int(e): g[cols].to_dict(orient="records") for e, g in test.groupby("engine")}, true_rul


@st.cache_data(show_spinner=False, ttl=600)
def predict(engine: int, history: list[dict]) -> dict:
    r = client().post("/predict", json={"engine_id": str(engine), "history": history})
    r.raise_for_status()
    return r.json()


# ---------- header + API check ----------
st.markdown("<h1>Fleet health <span class='badge'>DEMO · NASA C-MAPSS simulation data</span></h1>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Which engines need attention, and why. Simulation data stands in for real rotating "
            "equipment; one cycle ≈ one operating run.</div>", unsafe_allow_html=True)

try:
    health = client().get("/health", timeout=5).json()
except httpx.HTTPError:
    st.error(f"Can't reach the API at {API_URL}. Start it with `uvicorn api.main:app` or "
             "`docker run --env-file .env -p 8000:8000 pdm-case`.")
    st.stop()
if not health.get("models_loaded"):
    st.error("The API is up but has no models: run `python -m src.train`, then restart the API.")
    st.stop()

histories, true_rul = fleet_histories()
progress = st.progress(0.0, text="Scoring fleet…")
results = {}
for i, (eng, hist) in enumerate(histories.items(), 1):
    results[eng] = predict(eng, hist)
    progress.progress(i / len(histories), text=f"Scoring fleet… {i}/{len(histories)}")
progress.empty()

fleet = pd.DataFrame([{
    "engine": eng,
    "action": r["action"]["code"],
    "status": f"{STATUS[r['action']['code']]['icon']} {STATUS[r['action']['code']]['label']}",
    "status_label": STATUS[r["action"]["code"]]["label"],
    "rul": r["remaining_life"],
    "low": r["range"]["low"],
    "high": r["range"]["high"],
    "range": f"{r['range']['low']:.0f} – {r['range']['high']:.0f}",
    "alarm": r["early_warning"]["alarm"],
    "cycles": r["cycles_observed"],
    "true_rul": int(true_rul[eng]),
} for eng, r in results.items()])
fleet["order"] = fleet["action"].map({k: i for i, k in enumerate(STATUS)})
fleet = fleet.sort_values(["order", "low"]).reset_index(drop=True)
rules = next(iter(results.values()))["action"]
threshold, coverage = rules["replace_threshold"], next(iter(results.values()))["range"]["coverage"]

fleet_tab, engine_tab = st.tabs(["Fleet overview", "Engine detail"])

# ---------- fleet tab ----------
with fleet_tab:
    counts = fleet["action"].value_counts()
    cols = st.columns(4)
    kpi(cols[0], f"{STATUS['replace']['icon']} Replace now", str(counts.get("replace", 0)),
        f"lower bound ≤ {threshold} cycles", STATUS["replace"]["color"])
    kpi(cols[1], f"{STATUS['plan']['icon']} Plan", str(counts.get("plan", 0)),
        "early-warning alarm on: order parts, inspect", STATUS["plan"]["color"])
    kpi(cols[2], f"{STATUS['monitor']['icon']} Monitor", str(counts.get("monitor", 0)), "no action needed now",
        STATUS["monitor"]["color"])
    kpi(cols[3], "Median predicted life", f"{fleet['rul'].median():.0f}",
        f"cycles · {len(fleet)} engines · predictions capped near {CFG['data']['rul_cap']}", NEUTRAL)

    st.write("")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("##### Riskiest 15 engines: predicted life and its range")
        top = fleet.head(15).assign(label=lambda d: "Engine " + d["engine"].astype(str))
        y = alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False))
        color = alt.Color("status_label:N", scale=STATUS_SCALE, legend=alt.Legend(title=None, orient="bottom"))
        tip = [alt.Tooltip("label:N", title="Engine"), alt.Tooltip("status_label:N", title="Action"),
               alt.Tooltip("rul:Q", title="Predicted life", format=".0f"),
               alt.Tooltip("low:Q", title="Range low", format=".0f"), alt.Tooltip("high:Q", title="Range high", format=".0f")]
        rng = alt.Chart(top).mark_rule(strokeWidth=7, opacity=.32, strokeCap="round").encode(
            y=y, x=alt.X("low:Q", title=f"cycles left ({coverage:.0%} range)"), x2="high:Q", color=color, tooltip=tip)
        pts = alt.Chart(top).mark_circle(size=110, opacity=1, stroke="white", strokeWidth=1.5).encode(
            y=y, x="rul:Q", color=color, tooltip=tip)
        thr = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(strokeDash=[5, 4], color=STATUS["replace"]["color"]).encode(x="x:Q")
        st.altair_chart((rng + pts + thr).properties(height=480), width="stretch")
        st.caption(f"Dot = predicted remaining life; bar = {coverage:.0%} range; dashed line = replace threshold "
                   f"({threshold} cycles on the lower bound).")
    with right:
        st.markdown("##### All engines")
        f1, f2 = st.columns([2, 1])
        show = f1.pills("Show", [s["label"] for s in STATUS.values()], selection_mode="multi",
                        default=[s["label"] for s in STATUS.values()], label_visibility="collapsed")
        search = f2.text_input("Engine", placeholder="Find engine #", label_visibility="collapsed")
        view = fleet[fleet["status_label"].isin(show or [])]
        if search.strip().isdigit():
            view = view[view["engine"] == int(search)]
        st.dataframe(
            view[["engine", "status", "rul", "range", "alarm"]],
            hide_index=True, width="stretch", height=430,
            column_config={
                "engine": st.column_config.NumberColumn("Engine", format="%d", width="small"),
                "status": st.column_config.TextColumn("Action"),
                "rul": st.column_config.ProgressColumn("Predicted life", min_value=0, width="medium",
                                                       max_value=float(fleet["high"].max()), format="%.0f cycles"),
                "range": st.column_config.TextColumn("Range", width="small",
                                                     help=f"{coverage:.0%} range of remaining life (cycles)"),
                "alarm": st.column_config.CheckboxColumn("Warning", width="small"),
            })
        st.caption("Sorted by risk: action first, then lowest lower bound. Open an engine in Engine detail for its "
                   "health index, drivers and field note.")

# ---------- engine tab ----------
with engine_tab:
    eng = st.selectbox("Engine", fleet["engine"].tolist(),
                       format_func=lambda e: f"Engine {e} · {fleet.loc[fleet.engine == e, 'status'].iloc[0]}")
    r = results[eng]
    s = STATUS[r["action"]["code"]]
    st.markdown(f"<div class='banner' style='--c:{s['color']};--ink:{s['ink']}'><b>{s['icon']} {s['label']}</b>"
                f" &nbsp;·&nbsp; {html.escape(r['action']['text'].split(': ', 1)[-1])}</div>", unsafe_allow_html=True)
    st.write("")

    ew = r["early_warning"]
    cols = st.columns(4)
    kpi(cols[0], "Predicted life left", f"{r['remaining_life']:.0f}",
        f"cycles · {coverage:.0%} range {r['range']['low']:.0f}–{r['range']['high']:.0f}", s["color"])
    kpi(cols[1], "Early warning", "ON" if ew["alarm"] else "off",
        f"health index {ew['health_index']:.1f} vs threshold {ew['threshold']:.1f}",
        STATUS["plan"]["color"] if ew["alarm"] else NEUTRAL)
    kpi(cols[2], "Cycles run", str(r["cycles_observed"]), "since install", NEUTRAL)
    kpi(cols[3], "True life left", str(int(true_rul[eng])), "demo only: known in this benchmark", NEUTRAL)

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("##### Early-warning health index")
        base_cycles, persist = CFG["anomaly"]["baseline_cycles"], CFG["anomaly"]["persistence"]
        hi = pd.DataFrame({"cycle": range(1, len(r["health_index_history"]) + 1),
                           "health index": r["health_index_history"]})
        hover = alt.selection_point(nearest=True, on="pointerover", fields=["cycle"], empty=False)
        ref = alt.Chart(pd.DataFrame({"x0": [1], "x1": [base_cycles]})).mark_rect(opacity=.12, color=NEUTRAL).encode(
            x="x0:Q", x2="x1:Q")
        line = alt.Chart(hi).mark_line(strokeWidth=2.2, color=EXTENDS).encode(
            x=alt.X("cycle:Q", title="cycle"), y=alt.Y("health index:Q", title="health index (higher = worse)"))
        thr = alt.Chart(pd.DataFrame({"y": [ew["threshold"]]})).mark_rule(
            strokeDash=[5, 4], color=STATUS["plan"]["color"], strokeWidth=1.6).encode(y="y:Q")
        dots = line.mark_circle(size=70, color=EXTENDS).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("cycle:Q"), alt.Tooltip("health index:Q", format=".2f")]).add_params(hover)
        rule = alt.Chart(hi).mark_rule(color="gray", opacity=.4).encode(x="cycle:Q").transform_filter(hover)
        st.altair_chart((ref + line + thr + rule + dots).properties(height=320), width="stretch")
        st.caption(f"Distance of this engine's sensors from its own first {base_cycles} cycles (grey band), in noise "
                   f"units. Amber dashed line = alarm threshold; alarm after {persist} cycles in a row above it.")
    with right:
        st.markdown("##### Remaining life vs the replace threshold")
        span = pd.DataFrame({"low": [r["range"]["low"]], "high": [r["range"]["high"]], "rul": [r["remaining_life"]],
                             "y": ["life"]})
        xmax = max(140.0, r["range"]["high"] + 10)
        x = alt.X("low:Q", title="cycles left", scale=alt.Scale(domain=[0, xmax]))
        bar = alt.Chart(span).mark_rule(strokeWidth=16, opacity=.35, color=s["color"], strokeCap="round").encode(
            x=x, x2="high:Q", y=alt.Y("y:N", axis=None))
        dot = alt.Chart(span).mark_circle(size=260, color=s["color"], stroke="white", strokeWidth=2).encode(
            x="rul:Q", y=alt.Y("y:N", axis=None),
            tooltip=[alt.Tooltip("rul:Q", title="Predicted", format=".0f"), alt.Tooltip("low:Q", title="Low", format=".0f"),
                     alt.Tooltip("high:Q", title="High", format=".0f")])
        tline = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(
            strokeDash=[5, 4], color=STATUS["replace"]["color"], strokeWidth=2).encode(x="x:Q")
        st.altair_chart((bar + dot + tline).properties(height=110), width="stretch")
        st.caption(f"Bar = {coverage:.0%} range; dot = prediction; red dashed line = replace when the range's low end "
                   f"reaches {threshold} cycles (+{rules['lead_cycles']} cycles to replace).")

        st.markdown("##### What drives this estimate")
        drv = pd.DataFrame([{"driver": f"{d['name']} ({d['sensor']})" if d["sensor"] != "cycle" else d["name"],
                             "effect": d["effect_cycles"],
                             "direction": "shortens estimate" if d["effect_cycles"] < 0 else "extends estimate"}
                            for d in r["top_drivers"]])
        bars = alt.Chart(drv).mark_bar(cornerRadius=4).encode(
            y=alt.Y("driver:N", sort=None, title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
            x=alt.X("effect:Q", title="cycles vs the average engine"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["shortens estimate", "extends estimate"],
                                                         range=[SHORTENS, EXTENDS]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("driver:N"), alt.Tooltip("effect:Q", title="cycles", format="+.1f")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
        st.altair_chart((bars + zero).properties(height=alt.Step(44)), width="stretch")

    with st.expander("Driver details (plain language)"):
        for d in r["top_drivers"]:
            st.markdown(f"<div class='driver'>{html.escape(d['text'])}</div>", unsafe_allow_html=True)
        st.caption("SHAP pushes, relative to the average engine. They explain the model, not the physics.")

    st.markdown("##### Field note")
    if st.button("✍️ Write field note", key=f"note-{eng}", type="primary"):
        with st.spinner("Writing note…"):
            resp = client().post("/field-note", json={"engine_id": str(eng), "history": histories[eng]})
            resp.raise_for_status()
            fn = resp.json()["field_note"]
        # escaped: the note may come from an LLM, so never render it as HTML
        st.markdown(f"<div class='note'>{html.escape(fn['note'])}</div>", unsafe_allow_html=True)
        st.caption(f"Written by {fn['model']} (LLM); every number checked against the prediction."
                   if fn["source"] == "llm" else f"Template note ({fn['fallback_reason']}).")
