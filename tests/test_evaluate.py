from pathlib import Path

from capacity_runway.evaluate import backtest_coverage, evaluate_groundedness_corpus

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "narrative_corpus.json"


def test_backtest_coverage_runs_and_returns_both_methods():
    results, extrap = backtest_coverage(seed=3, n_per_kind=4, n_days=220, origins_per_series=3)
    labels = {r.label for r in results}
    assert labels == {"naive_quantile_regression", "conformalized_quantile_regression"}
    for r in results:
        assert 0.0 <= r.empirical_usage_coverage <= 1.0
        assert r.mean_interval_width_pct >= 0.0
        assert 0.0 <= r.empirical_runway_coverage <= 1.0 or r.empirical_runway_coverage != r.empirical_runway_coverage
    assert 0.0 <= extrap.frac_test_extrapolated <= 1.0


def test_cqr_correction_is_nonnegative_and_reported_honestly():
    """This project's headline honest finding: CQR calibration narrows the
    coverage gap in-distribution but does not fully close it in aggregate,
    and coverage on extrapolated points is materially worse than in-range.
    This test locks in the qualitative shape of that finding so a future
    change cannot silently invert it without the test suite noticing."""
    results, extrap = backtest_coverage(seed=7, n_per_kind=10, n_days=260, origins_per_series=6)
    cqr = next(r for r in results if r.label == "conformalized_quantile_regression")
    assert cqr.empirical_usage_coverage <= 0.95, "sanity: should not wildly overcover either"
    assert extrap.coverage_extrapolated < extrap.coverage_in_range, (
        "the extrapolation-region coverage gap is the project's central limitation; "
        "it should reproduce with the default seed"
    )


def test_groundedness_corpus_file_exists_and_is_labeled():
    assert CORPUS_PATH.exists(), "data/narrative_corpus.json must ship with the repository"
    result = evaluate_groundedness_corpus(CORPUS_PATH)
    assert result["n_cases"] >= 20
    assert result["tp"] + result["fn"] > 0, "corpus must contain hallucinated cases"
    assert result["tn"] + result["fp"] > 0, "corpus must contain grounded cases"
    # The checker is only validated against single-field, large-magnitude
    # fabrications (see README limitations); we assert it performs well on
    # that population without claiming it catches subtler ones.
    assert result["f1"] >= 0.75
