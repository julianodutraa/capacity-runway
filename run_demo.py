#!/usr/bin/env python3
"""Reproducible end-to-end demo: fit, calibrate, backtest, and narrate.

Run with: python run_demo.py
No API keys or network access are required; the LLM narrative step falls
back to a deterministic template when ANTHROPIC_API_KEY is not set.
"""
from __future__ import annotations

from pathlib import Path

from capacity_runway import features
from capacity_runway.data import generate_fleet
from capacity_runway.evaluate import backtest_coverage, evaluate_groundedness_corpus
from capacity_runway.forecast import CQRCalibrator, QuantileForecaster
from capacity_runway.narrative import render_llm_narrative, render_template_narrative
from capacity_runway.runway import compute_runway_interval

SEED = 7
THRESHOLD = 90.0


def main() -> None:
    print("=== Capacity Runway demo ===\n")

    print("[1/3] Backtesting naive quantile regression vs. conformalized quantile regression...")
    results, extrap = backtest_coverage(seed=SEED, threshold=THRESHOLD)
    for r in results:
        print(
            f"  {r.label:35s} nominal={r.nominal_coverage:.2f}  "
            f"usage_coverage={r.empirical_usage_coverage:.3f}  "
            f"mean_width={r.mean_interval_width_pct:5.2f}pp  "
            f"runway_coverage={r.empirical_runway_coverage:.3f}"
        )
    print(
        f"  [finding] {extrap.frac_test_extrapolated:.1%} of test points exceed every "
        f"utilization level seen in training; on those, coverage collapses to "
        f"{extrap.coverage_extrapolated:.1%} (vs {extrap.coverage_in_range:.1%} in-range) "
        f"with a {extrap.mean_bias_in_range_pp:+.2f}pp -> {extrap.mean_bias_extrapolated_pp:+.2f}pp "
        f"swing in mean bias, because tree-based quantile regressors cannot extrapolate."
    )

    print("\n[2/3] Forecasting one live example series and generating its narrative...")
    fleet = generate_fleet(n_per_kind=10, n_days=260, seed=SEED)
    example = next(s for s in fleet if s.kind == "bursty")
    train_frame = features.build_frame(fleet, (0.0, 0.50))
    cal_frame = features.build_frame(fleet, (0.50, 0.65))
    forecaster = QuantileForecaster.fit(train_frame.X, train_frame.y, seed=SEED)
    calibrator = CQRCalibrator.calibrate(forecaster, cal_frame.X, cal_frame.y, 0.90)

    origin = int(len(example.usage_pct) * 0.80)
    horizons = features.HORIZONS
    import numpy as np

    X = np.vstack([features._row_features(example.usage_pct, origin, h) for h in horizons])
    lo, mid, hi = calibrator.predict_interval(X)
    interval = compute_runway_interval(
        dict(zip(horizons, lo)), dict(zip(horizons, mid)), dict(zip(horizons, hi)), THRESHOLD
    )
    current_usage = float(example.usage_pct[origin])
    print(
        f"  series={example.series_id}  current_usage={current_usage:.1f}%  "
        f"runway=[{interval.runway_lo_days:.0f}, {interval.runway_hi_days:.0f}] days"
    )

    narrative = render_llm_narrative(example.series_id, current_usage, THRESHOLD, interval)
    source = "live LLM call"
    if narrative is None:
        narrative = render_template_narrative(example.series_id, current_usage, THRESHOLD, interval)
        source = "deterministic template (no ANTHROPIC_API_KEY set)"
    print(f"  narrative ({source}): {narrative}")

    print("\n[3/3] Scoring the groundedness checker against the cached labeled narrative corpus...")
    corpus_path = Path(__file__).resolve().parent / "data" / "narrative_corpus.json"
    if corpus_path.exists():
        g = evaluate_groundedness_corpus(corpus_path)
        print(
            f"  precision={g['precision']:.3f}  recall={g['recall']:.3f}  f1={g['f1']:.3f}  "
            f"accuracy={g['accuracy']:.3f}  (n={g['n_cases']})"
        )
    else:
        print("  data/narrative_corpus.json not found; skipping.")


if __name__ == "__main__":
    main()
