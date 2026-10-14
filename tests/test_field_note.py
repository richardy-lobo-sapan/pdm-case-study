from types import SimpleNamespace

import pytest

from src.field_note import check_numbers, facts, make_note, numbers_in, template_note

CFG = {"model": "openai/gpt-oss-20b", "reasoning_effort": "low", "temperature": 0.2,
       "max_completion_tokens": 600, "max_words": 80, "timeout_seconds": 15}

PRED = {  # shape of an /predict response
    "engine_id": "81", "cycles_observed": 213, "remaining_life": 7.2,
    "range": {"low": 0.4, "high": 61.4, "coverage": 0.9},
    "early_warning": {"alarm": True, "health_index": 9.1, "threshold": 2.32},
    "action": {"code": "replace", "text": "Replace now: the lower bound (0 cycles) is at or below the 5-cycle "
                                          "threshold; allow about 5 cycles to replace."},
    "top_drivers": [{"text": "LPT outlet temperature (s4) is 26.6 degR above this engine's healthy start; "
                             "this shortens the estimate by 23.0 cycles compared with the average engine."}],
}


class FakeLLM:
    """Stands in for the Groq client: returns a fixed note (or raises) and records the request."""
    def __init__(self, reply: str | Exception):
        self.reply, self.kwargs = reply, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def test_numbers_in_skips_sensor_names_and_reads_ranges():
    assert numbers_in("LPT outlet temperature (s4) rose") == []
    assert numbers_in("range 110-129, 90% sure, 26.6 degR") == [110, 129, 90, 26.6]


def test_template_note_uses_only_input_numbers():
    f = facts(PRED)
    note = template_note(f)
    assert check_numbers(note, f) == []
    assert "Replace now" in note and "Engine 81" in note


def test_check_catches_an_invented_number():
    assert check_numbers("Replace within 12 cycles.", facts(PRED)) == [12.0]
    assert check_numbers("About 7 cycles left (range 0 to 61), 23 cycles shorter.", facts(PRED)) == []  # 23 == 23.0


def test_no_key_means_template(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = make_note(PRED, CFG)
    assert out["source"] == "template" and out["fallback_reason"] == "GROQ_API_KEY not set"


def test_clean_llm_note_is_used_and_reasoning_not_requested():
    llm = FakeLLM("Engine 81 has about 7 cycles left (90% range 0 to 61) and an early-warning alarm, "
                  "mainly from LPT outlet temperature (s4). Replace now.")
    out = make_note(PRED, CFG, client=llm)
    assert out["source"] == "llm" and out["model"] == "openai/gpt-oss-20b"
    assert llm.kwargs["include_reasoning"] is False and llm.kwargs["model"] == CFG["model"]


@pytest.mark.parametrize("reply, reason", [
    ("Engine 81 should be replaced within 12 cycles.", "numbers not in the input"),
    (TimeoutError("slow"), "LLM call failed (TimeoutError)"),
    ("", "empty"),
    ("word " * 200, "too long"),
])
def test_bad_llm_output_falls_back_to_template(reply, reason):
    out = make_note(PRED, CFG, client=FakeLLM(reply))
    assert out["source"] == "template" and reason in out["fallback_reason"]
    assert check_numbers(out["note"], facts(PRED)) == []
