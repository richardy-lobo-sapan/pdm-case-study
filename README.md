# Predictive maintenance case study

[![tests](https://github.com/richardy-lobo-sapan/pdm-case-study/actions/workflows/tests.yml/badge.svg)](https://github.com/richardy-lobo-sapan/pdm-case-study/actions/workflows/tests.yml)

Can sensor data warn us early enough, and reliably enough, to save money on rotating equipment?
This project answers that end to end: from raw sensor readings to a cost-based maintenance decision,
served by an API, a dashboard and a short note for field engineers.

> **Stand-in data, assumed costs.** NASA's C-MAPSS turbofan simulation stands in for real rotating
> equipment (pumps, turbines, compressors); it is not real field data. All costs are placeholders in
> `configs/costs.yaml` and are labeled as assumptions everywhere they appear.

## The answer

- **Predictive maintenance costs 18% less per engine than the best fixed schedule** (95% range 9–25%),
  and 79% less than running to failure, under the assumed costs. Most of the saving comes from not
  throwing away parts that still had life left.
- **Early warning works with a simple method.** A health index that compares each engine with its own
  early cycles warned all 100 engines before failure, 90% of them at least 55 cycles ahead, with 4% false
  alarms. Fancier anomaly models did not beat it.
- **Remaining life comes with an honest range.** The model is off by about 13 cycles on unseen engines
  (RMSE 13.1, against 17.7 for a linear model given the same features), its calibrated 90% ranges held 89% of
  outcomes, and no test engine failed before its lower bound.
- **Where it stops paying:** if a breakdown costs less than 1.1× a planned job (downtime included), if wasted
  life is worth under $50 per cycle, or if replacing a part takes 20 or more cycles after an alert, the saving
  can no longer be told apart from zero.

The full story is in the deck, `deck/pdm_case_study.pptx`. Next to it, `deck/technical_review.pdf` is a
role-play technical review of the work (an AI reviewer acting as an industry data scientist), with the fixes it led to.

## How it works

- **Early warning:** each engine's sensors are compared with its own healthy start; an alarm fires when
  they drift past a threshold learned from healthy data.
- **Remaining life:** LightGBM on rolling-window features (level, trend, spread), trained on remaining life
  capped at 125 cycles. Ranges come from quantile models calibrated with cross-conformal prediction.
- **Explanations:** SHAP values from the model, summed per sensor, turned into the top drivers in plain words.
- **Cost-chosen alerts:** replace when the range's lower bound reaches 5 cycles. That threshold, and the
  best fixed schedule it is compared with, are chosen by a cost simulation, not by accuracy.
- **Deployment:** a FastAPI service (also in Docker), a Streamlit dashboard for daily triage, MLflow
  tracking, and a field note written by an LLM that may only use numbers from the model, with a fixed
  template as fallback.

All validation keeps each engine entirely on one side of a train/validation split, so no engine is ever
scored by a model that saw it.

## Honest limits

- Simulated data. The main results use FD001 (one fault mode, one operating condition); the harder FD004
  is a robustness check below. Real sensors are noisier.
- No early-life failures in this data (the shortest life is 128 cycles), which makes fixed schedules look
  safer than they would be in practice.
- Only 100 run-to-failure engines, so every result carries wide uncertainty.
- The LLM field note is checked number by number; its wording is not checked.

## Robustness checks

- **Harder data (FD004):** 6 operating conditions and 2 fault types, 249 training engines. With each sensor
  normalized within its operating condition, the same pipeline beats the best fixed schedule by 28% (95% range
  22–30%), and early warning still warned all 249 engines, 90% of them at least 44 cycles ahead.
- **FD004's weak spot is the range.** Error stays low for engines within 125 cycles of failure (RMSE 13.5; it is
  25.6 over all 248 test engines because 67 of them have more life than the 125-cycle training cap). But the
  calibrated ranges held 84% of test rows instead of 90%, and 15 test engines failed before their lower bound.
  Equipment that runs in several operating conditions needs ranges calibrated per condition before the lower
  bound can trigger maintenance.
- **The model reads wear, not just age:** without engine age as an input, FD001 test RMSE rises only from
  13.1 to 14.8.
- **Stable across random seeds:** test RMSE 13.0 ± 0.29 over 5 seeds.
- **The saving survives a stricter uncertainty check:** re-choosing the threshold inside every bootstrap
  resample gives a 95% range of 7–28%.

**Every number in the deck comes from code.** `deck/build_deck.py` registers each number with its source
in `deck/facts.json`, and `tests/test_deck.py` fails if the deck, the LinkedIn post or this README shows
any other number. (The check compares values, not meanings: it stops typed-in numbers, not a real number
reused in the wrong sentence.)

## Run it

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the NASA data (a small zip) and put `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in `data/raw/`
(and the three `FD004` files for the robustness check):

```text
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

Then:

```powershell
python -m src.train                  # full pipeline: results, models, MLflow run
python -m src.train --subset FD004   # robustness check on harder data (reports/results_FD004.json)
pytest -q                            # tests (also run on every push by GitHub Actions)
uvicorn api.main:app                 # API on http://127.0.0.1:8000
streamlit run dashboard/app.py       # dashboard (needs the API running)
python deck/build_deck.py            # rebuild the deck and LinkedIn post from the results
docker build -t pdm-case .
docker run --env-file .env -p 8000:8000 pdm-case
```

For the LLM field note, copy `.env.example` to `.env` and add a Groq API key; without one, the template note is used.
Notebooks in `notebooks/` use a kernel named `pdm-venv`; register it with
`python -m ipykernel install --sys-prefix --name pdm-venv`, or pick your own environment.

## What is where

| Path | What |
|---|---|
| `src/` | Data loading, features, early warning, remaining life, explanations, cost simulation, field note, full pipeline (`src/train.py`) |
| `api/main.py` | FastAPI service: `/health`, `/predict`, `/field-note` |
| `dashboard/app.py` | Streamlit dashboard: fleet triage and engine detail |
| `notebooks/` | Exploration and write-up for each step, with conclusions and limits |
| `configs/` | Settings (`config.yaml`) and assumed costs (`costs.yaml`) |
| `reports/` | Figures, `results.json` (single source of truth for every reported number) and `results_FD004.json` |
| `deck/` | The deck (`pdm_case_study.pptx`), the technical review (`technical_review.pdf`) and `build_deck.py`, which builds the deck from the results |
| `tests/` | Labels, engine-grouped splits, no look-ahead, cost math, API, field-note and deck number checks |
