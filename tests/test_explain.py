import re

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.explain import by_sensor, contributions, top_drivers

NAMES = {"s2": {"name": "LPC outlet temperature", "unit": "degR"},
         "s3": {"name": "HPC outlet temperature", "unit": "degR"},
         "cycle": {"name": "engine age (cycles run)", "unit": "cycles"}}


def _model_and_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"cycle": rng.integers(1, 300, 500).astype(float),
                      "s2_dev": rng.normal(size=500), "s2_mean5": rng.normal(size=500),
                      "s3_dev": rng.normal(size=500)})
    y = 100 - 0.2 * X.cycle - 15 * X.s2_dev - 5 * X.s2_mean5 + 3 * X.s3_dev
    return LGBMRegressor(n_estimators=50, verbose=-1, random_state=0).fit(X, y), X


def test_pushes_add_up_to_prediction():
    model, X = _model_and_data()
    c = contributions(model, X)
    np.testing.assert_allclose(c.sum(axis=1), model.predict(X), atol=1e-6)


def test_grouping_by_sensor_keeps_the_total():
    model, X = _model_and_data()
    c = contributions(model, X)
    g = by_sensor(c)
    assert sorted(g.columns) == ["cycle", "s2", "s3"]
    np.testing.assert_allclose(g.sum(axis=1), c.drop(columns="base").sum(axis=1), atol=1e-9)
    np.testing.assert_allclose(g["s2"], c["s2_dev"] + c["s2_mean5"])


def test_top_drivers_sorted_and_text_numbers_match_data():
    model, X = _model_and_data()
    drivers = top_drivers(model, X.iloc[[0]], NAMES, k=3)
    effects = [abs(d["effect_cycles"]) for d in drivers]
    assert effects == sorted(effects, reverse=True) and len(drivers) == 3
    for d in drivers:
        numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", d["text"].replace(d["sensor"], ""))}
        assert numbers == {abs(d["effect_cycles"]), abs(d["reading"])}
        assert ("shortens" in d["text"]) == (d["effect_cycles"] < 0)
