import numpy as np

from capacity_runway.data import generate_fleet, generate_series


def test_generate_series_shapes_and_bounds():
    s = generate_series("t-1", "linear", n_days=100, seed=1)
    assert len(s.usage_pct) == 100
    assert len(s.true_usage_pct) == 100
    assert np.all(s.usage_pct >= 0.0) and np.all(s.usage_pct <= 100.0)
    assert np.all(s.true_usage_pct >= 0.0) and np.all(s.true_usage_pct <= 100.0)


def test_linear_series_is_monotonic_on_average():
    s = generate_series("t-2", "linear", n_days=120, seed=2)
    # A linear growth trajectory's noise-free signal must be non-decreasing.
    assert np.all(np.diff(s.true_usage_pct) >= -1e-9)


def test_bursty_series_has_step_jumps():
    s = generate_series("t-3", "bursty", n_days=150, seed=3)
    diffs = np.diff(s.true_usage_pct)
    # At least one day-over-day jump should be much larger than the smooth trend slope.
    assert diffs.max() > 1.0


def test_generate_fleet_composition():
    fleet = generate_fleet(n_per_kind=4, n_days=90, seed=5)
    assert len(fleet) == 12
    kinds = {s.kind for s in fleet}
    assert kinds == {"linear", "seasonal", "bursty"}
    ids = [s.series_id for s in fleet]
    assert len(ids) == len(set(ids)), "series ids must be unique"


def test_deterministic_given_seed():
    a = generate_series("t-4", "seasonal", n_days=60, seed=99)
    b = generate_series("t-4", "seasonal", n_days=60, seed=99)
    np.testing.assert_array_equal(a.usage_pct, b.usage_pct)
