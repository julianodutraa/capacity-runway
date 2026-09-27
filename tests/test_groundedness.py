from capacity_runway.groundedness import check_narrative, extract_numeric_claims


def test_extract_numeric_claims_parses_units():
    claims = extract_numeric_claims("Usage is at 72% and will breach in 14-18 days.")
    values = [(v, u) for v, u, _ in claims]
    assert (72.0, "percent") in values
    assert (14.0, "day") in values
    assert (18.0, "day") in values


def test_check_narrative_grounded_case():
    text = "Resource x is at 67.5% utilization and should breach 90% in 12 to 18 days."
    report = check_narrative(
        text,
        current_usage_pct=67.5,
        threshold_pct=90.0,
        runway_lo_days=12,
        runway_hi_days=18,
    )
    assert not report.any_ungrounded
    assert report.n_claims >= 3


def test_check_narrative_flags_fabricated_day_count():
    # The narrative invents "45 days", far outside the true [12, 18] window.
    text = "Resource x is at 67.5% utilization and should breach 90% in about 45 days."
    report = check_narrative(
        text,
        current_usage_pct=67.5,
        threshold_pct=90.0,
        runway_lo_days=12,
        runway_hi_days=18,
    )
    assert report.any_ungrounded
    assert report.n_ungrounded >= 1


def test_check_narrative_flags_fabricated_percentage():
    text = "Resource x is currently at 82% utilization, well above the reported level."
    report = check_narrative(
        text,
        current_usage_pct=67.5,
        threshold_pct=90.0,
        runway_lo_days=12,
        runway_hi_days=18,
    )
    assert report.any_ungrounded


def test_check_narrative_tolerates_rounding():
    # 67.5 rounded to 68 should still be considered grounded within tolerance.
    text = "Utilization sits near 68% today."
    report = check_narrative(
        text,
        current_usage_pct=67.5,
        threshold_pct=90.0,
        runway_lo_days=12,
        runway_hi_days=18,
    )
    assert not report.any_ungrounded


def test_check_narrative_censored_runway_has_no_day_ground_truth():
    text = "No day count is mentioned here, only 55% utilization."
    report = check_narrative(
        text,
        current_usage_pct=55.0,
        threshold_pct=90.0,
        runway_lo_days=float("inf"),
        runway_hi_days=float("inf"),
    )
    assert not report.any_ungrounded
