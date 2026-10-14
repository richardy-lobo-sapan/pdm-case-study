"""Short plain-language note for a field engineer, from one /predict result.

An LLM (Groq) writes the note if GROQ_API_KEY is set; otherwise, or if the LLM fails the
number check (every number in the note must appear in the input facts), a fixed template is used.
So the returned note never contains a number the models did not produce.
"""
import json
import os
import re

# A number not glued to letters/digits/dots on its left: "s4" or "gpt-oss-20b" are names, not numbers.
NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?")


def facts(pred: dict) -> dict:
    """Only what an engineer needs, numbers pre-rounded for display so the LLM has no reason to round."""
    return {
        "engine": pred.get("engine_id") or "unknown",
        "cycles_run": pred["cycles_observed"],
        "remaining_life_cycles": round(pred["remaining_life"]),
        "range_low_cycles": round(pred["range"]["low"]),
        "range_high_cycles": round(pred["range"]["high"]),
        "range_confidence_percent": round(pred["range"]["coverage"] * 100),
        "early_warning_alarm": pred["early_warning"]["alarm"],
        "action": pred["action"]["code"],
        "action_text": pred["action"]["text"],
        "top_drivers": [d["text"] for d in pred["top_drivers"]],
    }


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in NUMBER.findall(text)]


def check_numbers(note: str, f: dict) -> list[float]:
    """Numbers in the note that do not appear in the facts (compared by value, so 23 matches 23.0)."""
    allowed = set(numbers_in(json.dumps(f)))
    return [n for n in numbers_in(note) if n not in allowed]


def template_note(f: dict) -> str:
    """Fixed-wording fallback: always available, uses only the facts."""
    alarm = ("ALARM: sensors have drifted from this engine's healthy start." if f["early_warning_alarm"]
             else "No early-warning alarm.")
    return (f"Engine {f['engine']}, after {f['cycles_run']} cycles: about {f['remaining_life_cycles']} cycles of life left "
            f"({f['range_confidence_percent']}% range {f['range_low_cycles']} to {f['range_high_cycles']}). {alarm} "
            f"Main reason: {f['top_drivers'][0]} {f['action_text']}")


def llm_note(f: dict, client, cfg: dict) -> str:
    """Ask the LLM for the note. Reasoning text is not requested; only the final answer is used."""
    system = ("You write short maintenance notes for field engineers, in plain language. "
              "Use ONLY numbers that appear in the facts, written exactly as given. Do not calculate, round, "
              "convert, count or add any other number. "
              f"Keep it under {cfg['max_words']} words. Say the remaining life with its range, whether the "
              "early-warning alarm is on or off, and the main driver. End with the action. "
              "Driver sensor numbers are CHANGES from this engine's healthy start, not absolute readings: always say "
              "'above/below its healthy start'. Driver effects are compared with an average engine: say so.")
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": "Facts (JSON):\n" + json.dumps(f, indent=1)}],
        temperature=cfg["temperature"],
        reasoning_effort=cfg["reasoning_effort"],
        include_reasoning=False,
        max_completion_tokens=cfg["max_completion_tokens"],
    )
    return (resp.choices[0].message.content or "").strip()


def make_note(pred: dict, cfg: dict, client=None) -> dict:
    """LLM note if possible and safe, else the template. Says which one and why.

    client: injectable for tests; by default a Groq client is created when GROQ_API_KEY is set.
    """
    f = facts(pred)
    fallback = lambda reason: {"note": template_note(f), "source": "template", "model": None, "fallback_reason": reason}
    if client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return fallback("GROQ_API_KEY not set")
        from groq import Groq  # only needed on the LLM path

        client = Groq(api_key=key, timeout=cfg["timeout_seconds"], max_retries=1)
    try:
        note = llm_note(f, client, cfg)
    except Exception as e:  # any network/API failure: the template is the specified fallback
        return fallback(f"LLM call failed ({type(e).__name__})")  # type only: never echo request details
    if not note:
        return fallback("LLM returned an empty note")
    if invented := check_numbers(note, f):
        return fallback(f"LLM used numbers not in the input: {invented}")
    if len(note.split()) > cfg["max_words"] * 1.5:
        return fallback("LLM note too long")
    return {"note": note, "source": "llm", "model": cfg["model"], "fallback_reason": None}
