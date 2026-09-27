"""Feature engineering for pooled multi-horizon quantile regression.

Rather than fitting one model per series (too little data per series for a
quantile regressor to be meaningful), we pool rolling-origin examples across
the whole synthetic fleet into a single global model. Each training example
describes "given what the series looked like up to day t, what will its
utilization be at day t + h", for a menu of forecast horizons h.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from capacity_runway.data import SyntheticSeries

HORIZONS: tuple[int, ...] = (7, 14, 21, 30, 45, 60)
_MIN_HISTORY = 21


@dataclasses.dataclass(frozen=True)
class Frame:
    X: np.ndarray
    y: np.ndarray
    series_ids: np.ndarray
    origin_days: np.ndarray
    horizons: np.ndarray
    feature_names: tuple[str, ...]


_FEATURE_NAMES = (
    "level",
    "slope_7d",
    "slope_21d",
    "volatility_14d",
    "dow_sin",
    "dow_cos",
    "horizon",
)


def _row_features(usage: np.ndarray, origin: int, horizon: int) -> np.ndarray:
    level = usage[origin]
    win7 = usage[max(0, origin - 7) : origin + 1]
    win21 = usage[max(0, origin - 21) : origin + 1]
    slope_7d = (win7[-1] - win7[0]) / max(1, len(win7) - 1)
    slope_21d = (win21[-1] - win21[0]) / max(1, len(win21) - 1)
    vol14 = usage[max(0, origin - 14) : origin + 1]
    volatility_14d = float(np.std(np.diff(vol14))) if len(vol14) > 2 else 0.0
    dow = (origin + horizon) % 7
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    return np.array(
        [level, slope_7d, slope_21d, volatility_14d, dow_sin, dow_cos, float(horizon)]
    )


def build_frame(
    fleet: list[SyntheticSeries],
    day_range: tuple[float, float],
    horizons: tuple[int, ...] = HORIZONS,
) -> Frame:
    """Build rolling-origin (X, y) pairs restricted to a fractional day range.

    ``day_range`` selects the slice of each series' timeline (as fractions of
    its length) from which origins may be drawn, which is how we carve each
    series into disjoint train / calibration / test windows in time order
    (never shuffled), avoiding look-ahead leakage across the split.
    """
    rows: list[np.ndarray] = []
    targets: list[float] = []
    sids: list[str] = []
    origins: list[int] = []
    hors: list[int] = []
    for s in fleet:
        n = len(s.usage_pct)
        lo_idx = int(n * day_range[0])
        hi_idx = int(n * day_range[1])
        for origin in range(max(lo_idx, _MIN_HISTORY), hi_idx):
            for h in horizons:
                target_idx = origin + h
                if target_idx >= n:
                    continue
                rows.append(_row_features(s.usage_pct, origin, h))
                targets.append(s.usage_pct[target_idx])
                sids.append(s.series_id)
                origins.append(origin)
                hors.append(h)
    return Frame(
        X=np.vstack(rows),
        y=np.array(targets),
        series_ids=np.array(sids),
        origin_days=np.array(origins),
        horizons=np.array(hors),
        feature_names=_FEATURE_NAMES,
    )
