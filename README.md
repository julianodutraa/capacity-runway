# Capacity Runway

Conformalized quantile regression for infrastructure capacity forecasting, paired with a fact-checked LLM narrative generator for the resulting risk report.

## Executive summary

Every data platform team eventually gets paged by a resource that ran out of room: a tablespace that hit 100%, a queue whose backlog outgrew its disk, a metrics store that filled its retention volume mid-incident. The usual mitigation is a dashboard with a static threshold alert, which fires only after the problem is already close, and a point forecast ("we will be full in 14 days") that carries no stated uncertainty, so nobody can tell whether it means "start a change request" or "page someone tonight".

This project builds the statistical layer that a capacity-planning tool needs before it can be trusted: a forecast of days-until-capacity-exhausted that comes with a calibrated confidence interval, not just a single number. It uses split conformal prediction, specifically Conformalized Quantile Regression (CQR, Romano et al., 2019), which is one of the few forecasting techniques with a distribution-free, finite-sample coverage guarantee: under its assumptions, a 90% interval really does contain the truth about 90% of the time, regardless of what the underlying growth process looks like.

The business case for this over an ad hoc forecast is simple. An interval that is honestly calibrated lets a team set a single, principled policy ("open a capacity ticket when the lower bound of the 90% runway interval drops below 21 days") instead of tuning a threshold alert by trial and error after every false alarm or missed warning. A narrower, well-calibrated interval also means fewer wasted engineering cycles chasing false alarms, and fewer surprises from an interval that looked safe but was silently miscalibrated.

This project also does not stop at the numbers. Capacity risk reports are increasingly drafted with an LLM in the loop, because a two-sentence narrative is what actually gets read in a Slack channel, not a JSON payload. That convenience creates a new failure mode: an LLM asked to summarize a forecast can quietly invent or misstate a number, and a fabricated "12 days" is more dangerous than no number at all, because it looks exactly as credible as a correct one. This repository ships a small, deterministic, dependency-free fact-checker that extracts every numeric claim from a generated narrative and verifies it against the numbers that were actually computed, and reports its precision and recall on a labeled corpus of LLM-written narratives, some of which were deliberately generated with a fabricated figure.

The honest headline result, stated up front rather than buried: conformal calibration does what it promises inside the range of utilization levels the model was trained on (89.1% empirical coverage against a 90% target), but degrades sharply once a resource grows past every level the model has ever seen (52.6% coverage on that slice). That is not a bug we hid; it is the central, reportable finding of this project, discussed in detail below and in [ARTICLE.md](ARTICLE.md).

## What is in this repository

```
capacity_runway/
  data.py          synthetic multi-series utilization generator (linear, seasonal, bursty)
  features.py      rolling-origin feature engineering, pooled across the fleet
  forecast.py      pooled gradient-boosted quantile regression + split conformal (CQR) calibration
  runway.py        converts a usage-level forecast band into a days-to-breach interval
  narrative.py     deterministic narrative template, plus an optional real-LLM path
  groundedness.py  regex-based numeric fact-checker for LLM-generated narratives
  evaluate.py      backtest harness: coverage audit, extrapolation-gap analysis, groundedness scoring
data/
  narrative_corpus.json   24 labeled LLM-generated narratives used to score the fact-checker
tests/             24 automated tests covering every module above
run_demo.py        end-to-end, reproducible demo script (no API key required)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 24 tests, no network access required
python run_demo.py           # fits, calibrates, backtests, forecasts, and narrates one example
```

Everything above runs entirely on synthetic data generated at runtime; there is no external dataset to download and no API key is required. If `ANTHROPIC_API_KEY` is set in the environment, `run_demo.py` will additionally call a real LLM to draft the narrative instead of using the deterministic template, and the same groundedness checker runs on whichever text was produced.

Sample output from `python run_demo.py`:

```
[1/3] Backtesting naive quantile regression vs. conformalized quantile regression...
  naive_quantile_regression           nominal=0.90  usage_coverage=0.821  mean_width=13.04pp  runway_coverage=0.867
  conformalized_quantile_regression   nominal=0.90  usage_coverage=0.852  mean_width=13.74pp  runway_coverage=0.867
  [finding] 10.7% of test points exceed every utilization level seen in training; on those, coverage collapses to
  52.6% (vs 89.1% in-range) with a -0.38pp -> +3.22pp swing in mean bias, because tree-based quantile regressors
  cannot extrapolate.

[3/3] Scoring the groundedness checker against the cached labeled narrative corpus...
  precision=1.000  recall=1.000  f1=1.000  accuracy=1.000  (n=24)
```

## Technical deep dive

### Problem setup

We generate a synthetic fleet of 30 daily utilization series (10 each of three archetypes: `linear`, `seasonal`, `bursty`) with a declared, noise-free ground-truth trajectory plus observation noise, so forecasts can be scored against a signal we actually know rather than only against one noisy realization. The `bursty` archetype models a common real failure mode that a smooth trend model misses entirely: long flat periods interrupted by step jumps, mimicking a backfill job or a retention policy change landing all at once.

### Forecasting method

For every series and every rolling origin day, we build a small feature vector (current level, 7-day and 21-day slope, 14-day volatility, day-of-week harmonics, and the forecast horizon) and pool these rolling-origin examples across the entire fleet into one global training set, split strictly in time order into train (0 to 50%), calibration (50% to 65%), and test (65% to 100%) windows per series, so no example ever uses information from its own future.

Three independent gradient-boosted quantile regressors (`sklearn.ensemble.GradientBoostingRegressor` with pinball loss, alpha in {0.05, 0.5, 0.95}) are trained on the pooled training window. This produces a naive, uncalibrated 90% band.

### Conformal calibration (CQR)

Following Romano, Patterson and Candes (2019), we compute a conformity score on the held-out calibration window, `max(lo(x) - y, y - hi(x))`, and inflate both sides of the interval by the `ceil((n+1)(1-alpha)) / n` empirical quantile of those scores. This is what gives CQR its finite-sample marginal coverage guarantee, and it is the reason the calibrated interval is never narrower than the naive one.

### The central honest finding: conformal calibration does not fix covariate shift

The guarantee above assumes the calibration and test examples are exchangeable. For a growing resource, they are not: by the time we reach the test window, utilization has grown past levels the calibration window ever saw. A gradient-boosted tree cannot extrapolate past the range of feature values seen in training, since every leaf outputs a constant learned from training examples; a resource that has grown past every previously observed level gets a prediction anchored to the nearest training leaf, which for a monotonically growing quantity is a systematic under-forecast.

We measured this directly. Splitting the test set by whether the "current level" feature exceeds the maximum level seen anywhere in training:

| Slice | Share of test points | Empirical coverage (target 90%) | Mean bias (actual minus predicted median) |
|---|---|---|---|
| In-range | 89.3% | 89.1% | -0.38 percentage points |
| Extrapolated | 10.7% | 52.6% | +3.22 percentage points |

Inside the training distribution, CQR delivers almost exactly what it promises. Outside it, coverage collapses to roughly half of the 90% target, and the bias flips sign and grows almost tenfold, in exactly the direction (under-forecasting) that would delay a capacity alert when it matters most. This is not a hyperparameter problem: it is a structural mismatch between a model family that cannot extrapolate and a forecasting task that is, by definition, always asking about levels not yet observed. The aggregate CQR coverage across the whole test set is 85.2%, short of the 90% target, entirely because of this 10.7% extrapolated slice pulling the average down.

We report the practical implication plainly rather than paper over it: this pooled tree-based CQR pipeline is more trustworthy the closer the forecast horizon is to already-observed levels, and its stated coverage should not be taken at face value once a resource is trending into genuinely new territory, which is exactly when a capacity forecast matters most. A production deployment should either (a) monitor and flag exactly this "extrapolated" condition and fall back to a wider, more conservative interval when it fires, or (b) use a model family with a linear or otherwise extrapolable trend component blended with the tree-based residual model. We did not implement either mitigation here, since diagnosing and quantifying the gap honestly is itself the point of this project; that write-up is scoped as a natural next step, not shipped as a hidden fix.

### Runway inversion

`runway.py` converts a per-horizon usage band into a "days until the threshold is breached" interval by finding the earliest horizon at which the upper bound crosses the threshold (the pessimistic, earliest-plausible-breach case) and the earliest horizon at which the lower bound crosses it (the optimistic, latest-plausible-breach case). Note that this derived interval does not inherit CQR's formal coverage guarantee, because a hitting time is a nonlinear function of the whole forecast path, not of a single horizon; instead of assuming the guarantee transfers, we validate it empirically. Across a held-out backtest of 180 (series, origin) pairs, the empirical runway-interval coverage was 86.7% for both the naive and the calibrated forecaster, close to but still short of the 90% nominal target, and, notably, not distinguishably improved by conformal calibration at this sample size. We report that flatly: on this metric, in this experiment, CQR's benefit did not clearly show up, which is a useful negative result about where conformal guarantees do and do not transfer.

### Grounded narrative generation

`narrative.py` ships a deterministic template renderer (no dependencies, cannot hallucinate by construction) and an optional real-LLM path used only if `ANTHROPIC_API_KEY` is set. `groundedness.py` is the fact-checking layer meant to sit between any LLM-drafted narrative and a human reader: it extracts every numeric claim in the text (handling ranges like "12 to 18 days" as well as single figures, and distinguishing day-counts from percentages so the two number families are never cross-matched) and flags any claim that does not fall within tolerance of one of the actually-computed ground-truth quantities.

We evaluated the checker against `data/narrative_corpus.json`, 24 narratives written by an independent LLM instance (a separate Claude subagent, blind to the checker's implementation) from 24 fixed scenarios; half were generated normally and half were generated with an explicit instruction to silently corrupt exactly one numeric field. On this corpus the checker reaches precision 1.000, recall 1.000, F1 1.000 (12 true positives, 12 true negatives, 0 false positives, 0 false negatives). We report the limitation of this result honestly: the corpus only tests single-field, large-magnitude fabrications (roughly 15 to 40 percent off, or several days off); a subtler hallucination that lands inside the numeric tolerance window would evade this checker by construction, since it is a threshold-based fact-checker, not a semantic one. That is a real limitation of the design, not an edge case we happened to miss in testing.

### Honest limitations, stated directly

The exchangeability assumption behind conformal prediction is violated twice over for this task: by ordinary temporal dependence between consecutive days, and, more consequentially as shown above, by covariate shift as a monotonically growing resource moves past its own training range. The runway-interval coverage (86.7%) falls short of its 90% target and was not clearly improved by calibration in this experiment. The groundedness checker is a syntactic, tolerance-based fact-checker, not a semantic one, and its perfect score on the shipped corpus reflects that corpus's large-magnitude fabrications rather than a claim of general hallucination detection. None of these numbers were adjusted after the fact to look better; they are exactly what `python run_demo.py` and `python -m pytest -q` reproduce.

## Read next

[ARTICLE.md](ARTICLE.md) is a narrative write-up of how this project came together, including the extrapolation finding above and what it implies for anyone building capacity forecasting on top of tree-based quantile regression.

## License

MIT, see [LICENSE](LICENSE).
