"""Translate usage-level forecasts into a business-facing runway interval.

Operators do not think in "utilization percent at horizon h"; they think in
"how many days until we run out". This module inverts a per-horizon usage
forecast band into a "days until the threshold is breached" interval, and
separately computes the ground-truth breach day from a noise-free synthetic
trajectory so we can audit whether the interval actually contains it.

Definition used here (stated explicitly because it matters for the coverage
audit in evaluate.py): given forecasts at horizons h_1 < h_2 < ... < h_k with
calibrated bands [lo(h), hi(h)],

    runway_lo = smallest h such that hi(h) >= threshold   (earliest plausible breach)
    runway_hi = smallest h such that lo(h) >= threshold   (latest plausible breach)

If hi(h) never reaches the threshold within the horizon menu, runway_lo is
reported as censored (">max horizon"); same for runway_hi with lo(h).
"""
from __future__ import annotations

import dataclasses

import numpy as np

CENSORED = float("inf")


@dataclasses.dataclass(frozen=True)
class RunwayInterval:
    runway_lo_days: float  # earliest plausible breach day (may be CENSORED)
    runway_hi_days: float  # latest plausible breach day (may be CENSORED)
    runway_mid_days: float  # point estimate from the median forecast
    max_horizon: int


def _first_crossing(values_by_horizon: dict[int, float], threshold: float) -> float:
    for h in sorted(values_by_horizon):
        if values_by_horizon[h] >= threshold:
            return float(h)
    return CENSORED


def compute_runway_interval(
    lo_by_horizon: dict[int, float],
    mid_by_horizon: dict[int, float],
    hi_by_horizon: dict[int, float],
    threshold: float,
) -> RunwayInterval:
    max_horizon = max(hi_by_horizon.keys())
    runway_lo = _first_crossing(hi_by_horizon, threshold)
    runway_hi = _first_crossing(lo_by_horizon, threshold)
    runway_mid = _first_crossing(mid_by_horizon, threshold)
    # By construction hi >= lo pointwise, so the earliest-breach search over hi(h)
    # can only find a crossing at or before the one over lo(h); guard anyway.
    if runway_hi < runway_lo:
        runway_hi = runway_lo
    return RunwayInterval(
        runway_lo_days=runway_lo,
        runway_hi_days=runway_hi,
        runway_mid_days=runway_mid,
        max_horizon=max_horizon,
    )


def true_crossing_day(
    true_usage: np.ndarray, origin: int, threshold: float, max_horizon: int
) -> float:
    """Ground-truth day (relative to origin) the noise-free trajectory first
    reaches ``threshold``, censored at ``max_horizon`` if it never does within
    the series or within the horizon menu."""
    end = min(len(true_usage), origin + max_horizon + 1)
    for day in range(origin, end):
        if true_usage[day] >= threshold:
            return float(day - origin)
    return CENSORED


def interval_contains(interval: RunwayInterval, truth_days: float) -> bool:
    """Whether the predicted runway interval covers the ground-truth breach day.

    Both a real, in-horizon breach and a correctly-censored "no breach in
    horizon" prediction count as covered.
    """
    if truth_days == CENSORED:
        return interval.runway_hi_days == CENSORED
    if interval.runway_lo_days == CENSORED:
        # predicted "no breach", but a breach actually happens: only counts as
        # covered if it happens strictly after the horizon we could see, which
        # can't be true since truth_days != CENSORED means it's within horizon.
        return False
    return interval.runway_lo_days <= truth_days <= interval.runway_hi_days
