"""A deterministic, regex-based fact-checker for numeric claims in LLM text.

This is not a semantic entailment model. It is deliberately the simplest
thing that can work: extract every number the narrative states, and confirm
each one is within tolerance of one of the numbers that were actually
computed (current usage, threshold, runway_lo, runway_hi, or an integer day
count anywhere inside the runway interval, since natural language often
paraphrases a range as one representative day). A number that matches
nothing in the ground-truth set is flagged as an ungrounded (possibly
hallucinated) claim.

The simplicity is the point: a lightweight, dependency-free checker that
runs on every LLM output before it reaches a human, catching the specific
and common failure mode of an LLM inventing a plausible-looking number, is
far more deployable than a second, larger LLM-as-judge call. Its precision
and recall on the labeled corpus in ``data/narrative_corpus.json`` are
reported honestly in ``evaluate.py``, including the cases it misses.
"""
from __future__ import annotations

import dataclasses
import re

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|days?|percent)?", re.IGNORECASE)

# A written range like "12 to 18 days" or "12-18 days" states the unit once,
# after the second number; a naive single-number scan would parse the first
# number of the pair as unitless and silently drop it from unit-aware
# checking. This range pattern is matched first so both endpoints inherit the
# trailing unit.
_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(%|days?|percent)\b", re.IGNORECASE
)

# A percentage figure and a "days" figure share the same numeric range in this
# domain (both are typically single- or double-digit), so the unit matters:
# without it, an ungrounded "72%" could be misjudged as matching a grounded
# "72 days" claim from a different scenario. We only cross-check within unit.
_DAY_TOKENS = {"day", "days"}
_PCT_TOKENS = {"%", "percent"}


@dataclasses.dataclass(frozen=True)
class NumericClaim:
    value: float
    unit: str  # "day", "percent", or "unitless"
    span: str
    grounded: bool


@dataclasses.dataclass(frozen=True)
class GroundednessReport:
    claims: tuple[NumericClaim, ...]
    any_ungrounded: bool

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def n_ungrounded(self) -> int:
        return sum(1 for c in self.claims if not c.grounded)


def _unit_name(raw_unit: str) -> str:
    unit_token = (raw_unit or "").strip().lower()
    if unit_token in _DAY_TOKENS:
        return "day"
    if unit_token in _PCT_TOKENS:
        return "percent"
    return "unitless"


def extract_numeric_claims(text: str) -> list[tuple[float, str, str]]:
    claims: list[tuple[float, str, str]] = []
    consumed = [False] * len(text)

    for match in _RANGE_RE.finditer(text):
        lo_raw, hi_raw, unit_raw = match.groups()
        unit = _unit_name(unit_raw)
        claims.append((float(lo_raw), unit, match.group(0).strip()))
        claims.append((float(hi_raw), unit, match.group(0).strip()))
        for i in range(*match.span()):
            consumed[i] = True

    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        if any(consumed[start:end]):
            continue
        raw_value, raw_unit = match.groups()
        claims.append((float(raw_value), _unit_name(raw_unit), match.group(0).strip()))

    return claims


def check_narrative(
    text: str,
    current_usage_pct: float,
    threshold_pct: float,
    runway_lo_days: float,
    runway_hi_days: float,
    day_tolerance: float = 1.0,
    pct_tolerance: float = 0.6,
) -> GroundednessReport:
    ground_truth_days = set()
    if runway_lo_days != float("inf") and runway_hi_days != float("inf"):
        lo_i, hi_i = int(round(runway_lo_days)), int(round(runway_hi_days))
        ground_truth_days = set(range(lo_i, hi_i + 1))
    ground_truth_pct = {round(current_usage_pct, 1), round(threshold_pct, 1)}

    raw_claims = extract_numeric_claims(text)
    claims: list[NumericClaim] = []
    for value, unit, span in raw_claims:
        if unit == "day":
            grounded = any(abs(value - d) <= day_tolerance for d in ground_truth_days)
        elif unit == "percent":
            grounded = any(abs(value - p) <= pct_tolerance for p in ground_truth_pct)
        else:
            # Unitless numbers are ambiguous by construction; check against
            # every ground-truth quantity with the looser of the two tolerances
            # rather than silently ignoring them (ignoring would inflate
            # precision by construction, which we explicitly want to avoid).
            candidates = ground_truth_pct | {float(d) for d in ground_truth_days}
            grounded = any(abs(value - c) <= max(day_tolerance, pct_tolerance) for c in candidates)
        claims.append(NumericClaim(value=value, unit=unit, span=span, grounded=grounded))

    return GroundednessReport(
        claims=tuple(claims),
        any_ungrounded=any(not c.grounded for c in claims),
    )
