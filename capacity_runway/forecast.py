"""Quantile regression forecaster and split conformal (CQR) calibration.

The statistical core of this project is Conformalized Quantile Regression
(CQR), following Romano, Patterson and Candes, "Conformalized Quantile
Regression" (NeurIPS 2019). CQR takes any pinball-loss quantile regressor
and, using a held-out calibration set disjoint from training, produces
prediction intervals with a finite-sample marginal coverage guarantee under
the exchangeability assumption. That assumption is not exactly true for a
time series (successive days are dependent, and the underlying growth
regime can shift), which is exactly the kind of gap a rigorous writeup has
to name rather than paper over; see ``evaluate.py`` for the empirical
coverage audit and the README/ARTICLE for the honest discussion.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

ALPHA_LOW = 0.05
ALPHA_HIGH = 0.95


def _fit_quantile_model(X: np.ndarray, y: np.ndarray, alpha: float, seed: int) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X, y)
    return model


@dataclasses.dataclass
class QuantileForecaster:
    """Pooled gradient-boosted quantile regressor for three quantiles."""

    lo_model: GradientBoostingRegressor
    mid_model: GradientBoostingRegressor
    hi_model: GradientBoostingRegressor

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "QuantileForecaster":
        return cls(
            lo_model=_fit_quantile_model(X, y, ALPHA_LOW, seed),
            mid_model=_fit_quantile_model(X, y, 0.5, seed + 1),
            hi_model=_fit_quantile_model(X, y, ALPHA_HIGH, seed + 2),
        )

    def predict_raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uncalibrated (naive) quantile predictions: lo, mid, hi."""
        lo = self.lo_model.predict(X)
        mid = self.mid_model.predict(X)
        hi = self.hi_model.predict(X)
        # Quantile crossing can happen with independently fit GBR heads;
        # enforce monotonicity, which is a standard, honestly-disclosed patch.
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        lo = np.minimum(lo, mid)
        hi = np.maximum(hi, mid)
        return lo, mid, hi


@dataclasses.dataclass
class CQRCalibrator:
    """Split conformal calibration on top of a fitted QuantileForecaster."""

    forecaster: QuantileForecaster
    correction: float
    target_coverage: float

    @classmethod
    def calibrate(
        cls,
        forecaster: QuantileForecaster,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        target_coverage: float = 0.90,
    ) -> "CQRCalibrator":
        lo, _, hi = forecaster.predict_raw(X_cal)
        conformity = np.maximum(lo - y_cal, y_cal - hi)
        n = len(y_cal)
        # Finite-sample corrected empirical quantile (Romano et al., 2019, eq. 6):
        # ceil((n+1)(1-alpha)) / n, clipped to 1.0 when n is small.
        alpha = 1.0 - target_coverage
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        correction = float(np.quantile(conformity, q_level, method="higher"))
        return cls(forecaster=forecaster, correction=correction, target_coverage=target_coverage)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrated (lo, mid, hi). Interval width grows by `correction` on each side."""
        lo, mid, hi = self.forecaster.predict_raw(X)
        return lo - self.correction, mid, hi + self.correction
