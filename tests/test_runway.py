import numpy as np

from capacity_runway.runway import (
    CENSORED,
    compute_runway_interval,
    interval_contains,
    true_crossing_day,
)


def test_compute_runway_interval_basic_crossing():
    horizons = [7, 14, 21, 30]
    lo = {7: 80.0, 14: 84.0, 21: 88.0, 30: 92.0}
    mid = {7: 83.0, 14: 87.0, 21: 91.0, 30: 95.0}
    hi = {7: 86.0, 14: 90.0, 21: 94.0, 30: 98.0}
    interval = compute_runway_interval(lo, mid, hi, threshold=90.0)
    assert interval.runway_lo_days == 14.0  # first horizon where hi >= 90
    assert interval.runway_hi_days == 30.0  # first horizon where lo >= 90
    assert interval.runway_mid_days == 21.0  # first horizon where mid >= 90


def test_compute_runway_interval_fully_censored():
    horizons = [7, 14]
    lo = {7: 40.0, 14: 42.0}
    mid = {7: 45.0, 14: 47.0}
    hi = {7: 50.0, 14: 52.0}
    interval = compute_runway_interval(lo, mid, hi, threshold=90.0)
    assert interval.runway_lo_days == CENSORED
    assert interval.runway_hi_days == CENSORED


def test_runway_hi_never_before_runway_lo():
    # Pathological input where lo(h) crosses before hi(h) due to a bug upstream
    # should still be clamped so the interval is never inverted.
    lo = {7: 95.0}
    mid = {7: 92.0}
    hi = {7: 91.0}
    interval = compute_runway_interval(lo, mid, hi, threshold=90.0)
    assert interval.runway_hi_days >= interval.runway_lo_days


def test_true_crossing_day_finds_first_breach():
    true_usage = np.array([50.0, 60.0, 70.0, 91.0, 95.0])
    day = true_crossing_day(true_usage, origin=0, threshold=90.0, max_horizon=10)
    assert day == 3.0


def test_true_crossing_day_censored_when_absent():
    true_usage = np.array([50.0, 55.0, 60.0])
    day = true_crossing_day(true_usage, origin=0, threshold=90.0, max_horizon=10)
    assert day == CENSORED


def test_interval_contains_covers_true_breach():
    horizons = [7, 14, 21]
    lo = {7: 80.0, 14: 88.0, 21: 92.0}
    mid = {7: 83.0, 14: 90.0, 21: 94.0}
    hi = {7: 86.0, 14: 92.0, 21: 96.0}
    interval = compute_runway_interval(lo, mid, hi, threshold=90.0)
    assert interval_contains(interval, truth_days=14.0)
    assert not interval_contains(interval, truth_days=6.0)


def test_interval_contains_agrees_on_no_breach():
    lo = {7: 40.0}
    mid = {7: 45.0}
    hi = {7: 50.0}
    interval = compute_runway_interval(lo, mid, hi, threshold=90.0)
    assert interval_contains(interval, truth_days=CENSORED)
    assert not interval_contains(interval, truth_days=5.0)
