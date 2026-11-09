# Reproduction and contract

Public results are aggregate evidence, not enough to recreate the licensed panel or fit a model. `results/provenance.json` records original generator hashes, table checksums, binding identity and principal library versions. Python 3.12 plus `requirements-research-lock.txt` records the original environment; this is not a complete transitive lock. The extracted package still needs a full actual-data reproduction gate.

Install the optional lakehouse extra. Obtain the real immutable binding from your lakehouse registry: snapshot_id, manifest_sha256, ordered feature_columns, optional cohort_extra_columns/signal_months, and features/cohort/targets entries with fully qualified table names and integer Delta versions. Do not invent versions. Authenticate through the Databricks SDK profile mechanism. The adapter uses the companion pinned GoldReader and paginated Statement transport; no operator IDs or secrets are shipped.

An existing authorized export can be prepared:

```bash
sp500-ml prepare --parquet data/private_gold.parquet --source-json data/source.json --output data/prepared
```

source.json contains provider=databricks_gold, the actual source_binding_sha256, and preferably snapshot_id/as_of. Manually supplied provenance is an operator claim; the adapter cannot authenticate it.

Rows require security_id, signal_date, ticker, sector, next_month_return, benchmark_return, target_top20, every feature group in src/sp500_ml/feature_groups.json,30 applicability flags and five quality flags. Dates are timezone-naive exchange dates, one per month. Security/ticker keys are unique within month. Labels/flags are binary, financials may be NULL, infinities and returns below −100% are rejected.

Latest outcomes must be unknown. Historical research requires known outcomes and complete technical/macro/sector inputs while retaining missing fundamentals. The command writes research/latest/population/candidates Parquets and checksums into a new empty directory. It preserves authoritative full-population labels; it does not re-rank after filtering. Gold's outcome-maturity policy remains authoritative.

EDA and modeling reruns write private outputs. Keep data/, outputs/ and artifacts/ out of Git. Gold feature construction and FMP extraction belong to the companion project.
