"""Synthetic generator for resource-utilization time series.

We do not have access to a real fleet of databases for this project, so we
generate synthetic daily utilization series (percentage of a fixed-size
resource consumed, for example a tablespace or an EBS-style volume) with
three archetypes that map to real capacity-planning failure modes:

* ``linear``    - steady, well-behaved organic growth (the easy case).
* ``seasonal``  - weekly seasonality (batch jobs on weekdays) layered on a
                  linear trend, which fools naive linear extrapolation.
* ``bursty``    - long flat periods punctuated by step jumps, which mimics a
                  backfill job or a retention-policy change landing at once.

Every series is generated from a declared ground-truth trajectory plus
observation noise, so we can score forecasts against a trajectory we
actually know, instead of only against one noisy realization.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

SeriesKind = Literal["linear", "seasonal", "bursty"]


@dataclasses.dataclass(frozen=True)
class SyntheticSeries:
    series_id: str
    kind: SeriesKind
    day: np.ndarray  # 0..n-1
    usage_pct: np.ndarray  # observed, noisy
    true_usage_pct: np.ndarray  # noise-free ground truth trajectory
    capacity_threshold_pct: float
    noise_std: float


def _linear_trajectory(n_days: int, rng: np.random.Generator) -> np.ndarray:
    start = rng.uniform(35.0, 55.0)
    slope = rng.uniform(0.12, 0.35)
    t = np.arange(n_days)
    return start + slope * t


def _seasonal_trajectory(n_days: int, rng: np.random.Generator) -> np.ndarray:
    start = rng.uniform(35.0, 55.0)
    slope = rng.uniform(0.08, 0.22)
    amplitude = rng.uniform(2.0, 5.0)
    t = np.arange(n_days)
    weekly = amplitude * np.sin(2 * np.pi * t / 7.0 + rng.uniform(0, 2 * np.pi))
    # weekday batch jobs push usage up on business days, weekends relax it
    weekday_bump = np.where((t % 7) < 5, amplitude * 0.4, -amplitude * 0.4)
    return start + slope * t + weekly + weekday_bump


def _bursty_trajectory(n_days: int, rng: np.random.Generator) -> np.ndarray:
    start = rng.uniform(30.0, 45.0)
    slope = rng.uniform(0.03, 0.10)
    t = np.arange(n_days)
    trend = start + slope * t
    n_jumps = rng.integers(2, 5)
    jump_days = np.sort(rng.choice(np.arange(10, n_days - 5), size=n_jumps, replace=False))
    jump_sizes = rng.uniform(3.0, 9.0, size=n_jumps)
    steps = np.zeros(n_days)
    for jd, js in zip(jump_days, jump_sizes):
        steps[jd:] += js
    return trend + steps


_GENERATORS = {
    "linear": _linear_trajectory,
    "seasonal": _seasonal_trajectory,
    "bursty": _bursty_trajectory,
}


def generate_series(
    series_id: str,
    kind: SeriesKind,
    n_days: int = 180,
    noise_std: float = 1.1,
    capacity_threshold_pct: float = 90.0,
    seed: int = 0,
) -> SyntheticSeries:
    """Generate one synthetic utilization series with a known ground truth."""
    if kind not in _GENERATORS:
        raise ValueError(f"unknown series kind: {kind!r}")
    rng = np.random.default_rng(seed)
    true_usage = _GENERATORS[kind](n_days, rng)
    true_usage = np.clip(true_usage, 0.0, 100.0)
    noise = rng.normal(0.0, noise_std, size=n_days)
    observed = np.clip(true_usage + noise, 0.0, 100.0)
    return SyntheticSeries(
        series_id=series_id,
        kind=kind,
        day=np.arange(n_days),
        usage_pct=observed,
        true_usage_pct=true_usage,
        capacity_threshold_pct=capacity_threshold_pct,
        noise_std=noise_std,
    )


def generate_fleet(n_per_kind: int = 8, n_days: int = 180, seed: int = 42) -> list[SyntheticSeries]:
    """Generate a small synthetic fleet spanning all three growth archetypes."""
    rng = np.random.default_rng(seed)
    fleet: list[SyntheticSeries] = []
    counter = 0
    for kind in ("linear", "seasonal", "bursty"):
        for _ in range(n_per_kind):
            series_seed = int(rng.integers(0, 1_000_000))
            noise_std = float(rng.uniform(0.6, 1.8))
            series_id = f"{kind}-{counter:03d}"
            fleet.append(
                generate_series(
                    series_id=series_id,
                    kind=kind,  # type: ignore[arg-type]
                    n_days=n_days,
                    noise_std=noise_std,
                    seed=series_seed,
                )
            )
            counter += 1
    return fleet
