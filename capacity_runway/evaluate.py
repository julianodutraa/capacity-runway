"""End-to-end evaluation harness: forecast backtest + groundedness audit.

Two independent evaluations live here, matching the two claims this project
makes:

1. ``backtest_coverage`` measures, on held-out data the models never trained
   or calibrated on, whether CQR's calibrated intervals actually achieve
   their nominal coverage, both at the raw usage level (the level CQR's
   theory covers) and at the derived runway-day level (the level operators
   actually care about, which the theory does not directly cover). It
   reports the naive (uncalibrated) baseline side by side, honestly,
   including where CQR does not clearly beat it.

2. ``evaluate_groundedness_corpus`` scores the numeric fact-checker in
   ``groundedness.py`` against a labeled corpus of LLM-generated narratives,
   reporting precision/recall/F1 for detecting the ones that were
   deliberately generated with a fabricated number.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from capacity_runway import features
from capacity_runway.data import generate_fleet
from capacity_runway.forecast import CQRCalibrator, QuantileForecaster
from capacity_runway.groundedness import check_narrative
from capacity_runway.runway import compute_runway_interval, interval_contains, true_crossing_day

TRAIN_RANGE = (0.0, 0.50)
CAL_RANGE = (0.50, 0.65)
TEST_RANGE = (0.65, 1.0)


@dataclasses.dataclass
class CoverageResult:
    label: str
    nominal_coverage: float
    empirical_usage_coverage: float
    mean_interval_width_pct: float
    empirical_runway_coverage: float
    n_usage_points: int
    n_runway_points: int


def _usage_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    covered = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(covered)), float(np.mean(hi - lo))


def _runway_backtest(
    fleet, predict_fn, threshold: float, horizons: tuple[int, ...], origins_per_series: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    covered = []
    max_h = max(horizons)
    for s in fleet:
        n = len(s.usage_pct)
        test_lo = int(n * TEST_RANGE[0])
        test_hi = n - max_h - 1
        if test_hi <= test_lo:
            continue
        origins = rng.integers(test_lo, test_hi, size=origins_per_series)
        for origin in origins:
            row_by_h = {h: features._row_features(s.usage_pct, int(origin), h) for h in horizons}
            X = np.vstack([row_by_h[h] for h in horizons])
            lo, mid, hi = predict_fn(X)
            lo_by_h = dict(zip(horizons, lo))
            mid_by_h = dict(zip(horizons, mid))
            hi_by_h = dict(zip(horizons, hi))
            interval = compute_runway_interval(lo_by_h, mid_by_h, hi_by_h, threshold)
            truth = true_crossing_day(s.true_usage_pct, int(origin), threshold, max_h)
            covered.append(interval_contains(interval, truth))
    return float(np.mean(covered)) if covered else float("nan")


@dataclasses.dataclass
class ExtrapolationBreakdown:
    train_max_level: float
    frac_test_extrapolated: float
    coverage_extrapolated: float
    coverage_in_range: float
    mean_bias_extrapolated_pp: float
    mean_bias_in_range_pp: float
    n_extrapolated: int
    n_in_range: int


def extrapolation_breakdown(
    fleet, train_frame: "features.Frame", test_frame: "features.Frame", calibrator: CQRCalibrator
) -> ExtrapolationBreakdown:
    """Split test-set coverage by whether the "current level" feature exceeds
    anything the model saw during training.

    Gradient-boosted trees predict a constant value per leaf, so they cannot
    extrapolate past the range of covariates seen in training: a resource
    that has grown past every level observed during training gets a
    prediction clamped to the nearest training leaf, which is a systematic
    under-forecast for a monotonically growing quantity. This is a form of
    covariate shift that split conformal calibration does not fix, because
    the calibration set is drawn from the same (pre-shift) distribution as
    training. It is the central limitation of this project, documented and
    measured rather than hidden.
    """
    train_max_level = float(train_frame.X[:, 0].max())
    lo, mid, hi = calibrator.predict_interval(test_frame.X)
    covered = (test_frame.y >= lo) & (test_frame.y <= hi)
    extrap = test_frame.X[:, 0] > train_max_level
    bias = test_frame.y - mid
    return ExtrapolationBreakdown(
        train_max_level=train_max_level,
        frac_test_extrapolated=float(np.mean(extrap)),
        coverage_extrapolated=float(np.mean(covered[extrap])) if extrap.any() else float("nan"),
        coverage_in_range=float(np.mean(covered[~extrap])) if (~extrap).any() else float("nan"),
        mean_bias_extrapolated_pp=float(np.mean(bias[extrap])) if extrap.any() else float("nan"),
        mean_bias_in_range_pp=float(np.mean(bias[~extrap])) if (~extrap).any() else float("nan"),
        n_extrapolated=int(np.sum(extrap)),
        n_in_range=int(np.sum(~extrap)),
    )


def backtest_coverage(
    seed: int = 7,
    n_per_kind: int = 10,
    n_days: int = 260,
    threshold: float = 90.0,
    target_coverage: float = 0.90,
    origins_per_series: int = 6,
) -> tuple[list[CoverageResult], ExtrapolationBreakdown]:
    fleet = generate_fleet(n_per_kind=n_per_kind, n_days=n_days, seed=seed)
    train_frame = features.build_frame(fleet, TRAIN_RANGE)
    cal_frame = features.build_frame(fleet, CAL_RANGE)
    test_frame = features.build_frame(fleet, TEST_RANGE)

    forecaster = QuantileForecaster.fit(train_frame.X, train_frame.y, seed=seed)
    calibrator = CQRCalibrator.calibrate(forecaster, cal_frame.X, cal_frame.y, target_coverage)

    results = []
    for label, predict_fn in (
        ("naive_quantile_regression", forecaster.predict_raw),
        ("conformalized_quantile_regression", calibrator.predict_interval),
    ):
        lo, mid, hi = predict_fn(test_frame.X)
        usage_cov, mean_width = _usage_coverage(test_frame.y, lo, hi)
        runway_cov = _runway_backtest(
            fleet, predict_fn, threshold, features.HORIZONS, origins_per_series, seed=seed + 1
        )
        results.append(
            CoverageResult(
                label=label,
                nominal_coverage=target_coverage,
                empirical_usage_coverage=usage_cov,
                mean_interval_width_pct=mean_width,
                empirical_runway_coverage=runway_cov,
                n_usage_points=len(test_frame.y),
                n_runway_points=n_per_kind * 3 * origins_per_series,
            )
        )
    extrap = extrapolation_breakdown(fleet, train_frame, test_frame, calibrator)
    return results, extrap


def evaluate_groundedness_corpus(corpus_path: Path) -> dict:
    corpus = json.loads(corpus_path.read_text())
    tp = fp = tn = fn = 0
    per_case = []
    for case in corpus["cases"]:
        report = check_narrative(
            text=case["narrative"],
            current_usage_pct=case["current_usage_pct"],
            threshold_pct=case["threshold_pct"],
            runway_lo_days=case["runway_lo_days"],
            runway_hi_days=case["runway_hi_days"],
        )
        predicted_hallucinated = report.any_ungrounded
        actual_hallucinated = case["is_hallucinated"]
        if predicted_hallucinated and actual_hallucinated:
            tp += 1
        elif predicted_hallucinated and not actual_hallucinated:
            fp += 1
        elif not predicted_hallucinated and actual_hallucinated:
            fn += 1
        else:
            tn += 1
        per_case.append(
            {
                "id": case["id"],
                "actual_hallucinated": actual_hallucinated,
                "predicted_hallucinated": predicted_hallucinated,
                "n_claims": report.n_claims,
                "n_ungrounded": report.n_ungrounded,
            }
        )
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "n_cases": len(corpus["cases"]),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "per_case": per_case,
        "provenance": corpus.get("provenance"),
    }


if __name__ == "__main__":
    import sys

    results, extrap = backtest_coverage()
    for r in results:
        print(
            f"{r.label}: nominal={r.nominal_coverage:.2f} "
            f"usage_coverage={r.empirical_usage_coverage:.3f} "
            f"mean_width={r.mean_interval_width_pct:.2f}pp "
            f"runway_coverage={r.empirical_runway_coverage:.3f} "
            f"(n_usage={r.n_usage_points}, n_runway={r.n_runway_points})"
        )
    print(
        f"\nextrapolation breakdown: train_max_level={extrap.train_max_level:.1f}% "
        f"frac_test_extrapolated={extrap.frac_test_extrapolated:.3f} "
        f"coverage_in_range={extrap.coverage_in_range:.3f} "
        f"coverage_extrapolated={extrap.coverage_extrapolated:.3f} "
        f"mean_bias_in_range={extrap.mean_bias_in_range_pp:+.2f}pp "
        f"mean_bias_extrapolated={extrap.mean_bias_extrapolated_pp:+.2f}pp"
    )

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "narrative_corpus.json"
    if corpus_path.exists():
        g = evaluate_groundedness_corpus(corpus_path)
        print(
            f"\ngroundedness checker: precision={g['precision']:.3f} recall={g['recall']:.3f} "
            f"f1={g['f1']:.3f} accuracy={g['accuracy']:.3f} "
            f"(tp={g['tp']} fp={g['fp']} tn={g['tn']} fn={g['fn']}, n={g['n_cases']})"
        )
    else:
        print("\nnarrative_corpus.json not found, skipping groundedness evaluation", file=sys.stderr)
