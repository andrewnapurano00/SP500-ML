# Data and methodology

**WORK IN PROGRESS.** The hypothesis is incremental fundamental ranking information over a technical/sector baseline. Membership, identity, dates and eligibility stay fixed in matched comparisons.

## Data and labels

The lakehouse owns FMP ingestion, PostgreSQL history, availability-aware features and immutable Delta snapshots. This package consumes explicit Gold bindings and private checksummed exports. It has no source-API collector or PostgreSQL fallback. Availability rules do not prove that today's vendor record equals the original filing vintage.

The mature research filter requires known stock/VOO next-month returns, a top-quintile label and complete technical/macro/sector context. Missing fundamentals do not remove stocks. The latest feature-only cohort retains unknown outcomes. Top-quintile labels rank the full retained monthly membership population before research filtering. Beating VOO is a distinct label. Targets never enter the estimator.

## Preprocessing and experiments

Stock inputs use within-month percentile ranks over the complete eligible cohort. Sector means/scales use unique training sector-months. Financial scores use fixed transformations. Raw financial variants fit clipping, centering and scaling inside each training fold. Missing logistic contributions are zero with separate observed/applicable gates; warehouse values remain NULL. Boosted raw inputs retain native NaN.

Twelve variants compare sector technicals, fundamentals-only, score/raw/combined representations, a boosted combination, four financial families and two beats-VOO variants. Logistic regression uses C=1, seed42, max_iter2000 and convergence warnings as errors. XGBoost uses depth2, min_child_weight50, learning rate0.05, lambda20 and seed42. Code is authoritative.

Each test month follows 36 training months and one purged month. Boosting rounds40/80/120 are selected inside each outer training period: 29 inner training months, one purge, six validation months. Preprocessing is refitted inside the inner split. The original run used897 fits including inner and latest feature-only fits. The unknown future outcome is excluded from 68 evaluated return months.

## Evaluation and inference

Monthly AUC, average precision, log loss and Brier measure classification quality. Confusion matrices compare actual top-quintile labels with the top-30 selection decision. Portfolio selection ranks probabilities, takes30 names with deterministic ticker tie-breaking, and equally weights them at zero trading costs. VOO uses identical return months. Drawdown includes initial wealth1; annual tables mark partial years. The private frozen capped-allocation policy is a different strategy.

Paired circular block bootstrap uses 5,000 resamples, three-month blocks and six-month sensitivity. EDA jointly adjusts association and quintile-spread endpoints using BH and reports BY. Pointwise intervals are not multiplicity-adjusted. Macro inference uses one observation per month; sector inference uses sector-months; stock rank inference aggregates within month. This avoids treating repeated market values as independent stock evidence.

The history has influenced development and is not a fresh confirmatory holdout. Financial revisions, classifications, delistings, execution timing and solver stability remain limitations. A single-stock scoring API lacks the full cohort needed for percentile ranks.
