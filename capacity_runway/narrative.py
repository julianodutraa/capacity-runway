"""Turn a numeric runway forecast into a short natural-language report.

Two generators are provided:

``render_template_narrative``
    A deterministic, dependency-free renderer that interpolates the computed
    numbers directly into a fixed sentence template. It cannot hallucinate,
    by construction, and is what the automated test suite and the CLI demo
    use by default so the project has zero required external dependencies.

``render_llm_narrative`` (optional)
    A thin wrapper that calls a real LLM (Anthropic's API, if
    ``ANTHROPIC_API_KEY`` is set) to write a more natural paragraph from the
    same numbers. Free-text generation from an LLM is exactly where
    fabricated figures creep in, which is the failure mode ``groundedness.py``
    is built to catch. The evaluation shipped in this repository was produced
    by an independent LLM (see ``data/narrative_corpus.json`` for the
    generation provenance) so the checker is graded on text it did not write
    itself.
"""
from __future__ import annotations

import os

from capacity_runway.runway import CENSORED, RunwayInterval


def _fmt_days(value: float) -> str:
    if value == CENSORED:
        return "beyond the forecast horizon"
    return f"{value:.0f} days"


def render_template_narrative(
    series_id: str,
    current_usage_pct: float,
    threshold_pct: float,
    interval: RunwayInterval,
) -> str:
    if interval.runway_lo_days == CENSORED:
        breach_clause = (
            f"is not expected to reach the {threshold_pct:.0f}% threshold within the "
            f"{interval.max_horizon}-day forecast horizon"
        )
    else:
        breach_clause = (
            f"is projected to reach the {threshold_pct:.0f}% threshold in "
            f"{_fmt_days(interval.runway_lo_days)} to {_fmt_days(interval.runway_hi_days)}"
        )
    return (
        f"Resource {series_id} is currently at {current_usage_pct:.1f}% utilization and "
        f"{breach_clause}, at the 90% conformal confidence level."
    )


def render_llm_narrative(
    series_id: str,
    current_usage_pct: float,
    threshold_pct: float,
    interval: RunwayInterval,
) -> str | None:
    """Best-effort real-LLM narrative. Returns None if no API key is configured
    or the call fails, so callers should fall back to the template renderer."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Write a two-sentence executive capacity-risk note for a data "
            f"infrastructure resource named {series_id}, currently at "
            f"{current_usage_pct:.1f}% of capacity, capacity threshold "
            f"{threshold_pct:.0f}%, projected to breach the threshold between "
            f"{_fmt_days(interval.runway_lo_days)} and {_fmt_days(interval.runway_hi_days)} "
            "from now. Use only these numbers, do not invent any other figures."
        )
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:
        return None
