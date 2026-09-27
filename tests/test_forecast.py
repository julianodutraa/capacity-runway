import numpy as np

from capacity_runway import features
from capacity_runway.data import generate_fleet
from capacity_runway.forecast import CQRCalibrator, QuantileForecaster


def _small_fleet():
    return generate_fleet(n_per_kind=6, n_days=220, seed=11)


def test_quantile_forecaster_orders_lo_mid_hi():
    fleet = _small_fleet()
    train = features.build_frame(fleet, (0.0, 0.5))
    test = features.build_frame(fleet, (0.65, 1.0))
    forecaster = QuantileForecaster.fit(train.X, train.y, seed=11)
    lo, mid, hi = forecaster.predict_raw(test.X)
    assert np.all(lo <= mid + 1e-9)
    assert np.all(mid <= hi + 1e-9)


def test_cqr_calibration_widens_or_keeps_interval():
    fleet = _small_fleet()
    train = features.build_frame(fleet, (0.0, 0.5))
    cal = features.build_frame(fleet, (0.5, 0.65))
    test = features.build_frame(fleet, (0.65, 1.0))
    forecaster = QuantileForecaster.fit(train.X, train.y, seed=11)
    raw_lo, _, raw_hi = forecaster.predict_raw(test.X)

    calibrator = CQRCalibrator.calibrate(forecaster, cal.X, cal.y, target_coverage=0.90)
    cal_lo, _, cal_hi = calibrator.predict_interval(test.X)

    assert calibrator.correction >= 0.0
    # CQR can only widen (or leave unchanged) the raw interval, never narrow it,
    # since correction >= 0 by construction (it is a quantile of nonnegative
    # conformity scores... unless the raw interval already fully contains the
    # calibration residuals, in which case correction could still be 0).
    assert np.all(cal_lo <= raw_lo + 1e-9)
    assert np.all(cal_hi >= raw_hi - 1e-9)


def test_cqr_improves_or_matches_coverage_over_naive():
    """The central statistical claim of CQR: calibrated coverage should be at
    least as close to nominal as the naive baseline, on average, on held-out
    data. We check it holds in aggregate (not per-point, which CQR does not
    promise)."""
    fleet = generate_fleet(n_per_kind=10, n_days=260, seed=7)
    train = features.build_frame(fleet, (0.0, 0.50))
    cal = features.build_frame(fleet, (0.50, 0.65))
    test = features.build_frame(fleet, (0.65, 1.0))
    forecaster = QuantileForecaster.fit(train.X, train.y, seed=7)
    calibrator = CQRCalibrator.calibrate(forecaster, cal.X, cal.y, target_coverage=0.90)

    raw_lo, _, raw_hi = forecaster.predict_raw(test.X)
    cal_lo, _, cal_hi = calibrator.predict_interval(test.X)

    naive_cov = np.mean((test.y >= raw_lo) & (test.y <= raw_hi))
    cqr_cov = np.mean((test.y >= cal_lo) & (test.y <= cal_hi))

    naive_gap = abs(0.90 - naive_cov)
    cqr_gap = abs(0.90 - cqr_cov)
    assert cqr_gap <= naive_gap + 0.02, (
        f"CQR should not be meaningfully worse than the naive baseline in aggregate "
        f"(naive_cov={naive_cov:.3f}, cqr_cov={cqr_cov:.3f})"
    )
