# Deployment preparation

**WORK IN PROGRESS · LOCAL_RESEARCH_CANDIDATE.** GitHub publication is complete when this repository is online; it is not a hosted prediction service. Frozen M03 remains a separate artifact/policy.

## Implemented batch path

`train-candidate` fits financial-strength logistic regression on 36 contiguous mature months ending two months before the signal month, leaving one purge month. It saves a private estimator and manifest: preprocessing state, input order, cutoffs, Gold binding, dependency/source hashes and exact eligible-cohort digest. It checks save/reload probability equality.

```bash
sp500-ml train-candidate --input data/prepared --signal-month 2026-10 --bundle artifacts/candidate-2026-10
# Store the printed manifest digest through a trusted operator channel.
sp500-ml score-candidate --bundle artifacts/candidate-2026-10 --manifest-sha256 TRUSTED_MANIFEST_SHA256 --cohort data/prepared/latest_with_fundamentals.parquet --output outputs/scores-2026-10.parquet
```

Load only trusted joblib artifacts. A checksum verifies expected bytes; it does not make untrusted serialized code safe. Artifacts/scores stay private and ignored by Git. The scorer checks dependency/source versions, model bytes, feature order, signal month, unique IDs and the complete pinned cohort. Full-cohort ranks preclude isolated single-stock requests.

The candidate supports the balance-sheet score specification, not every raw/boosted experiment. It emits probabilities, not orders. Synthetic CI is not actual-Gold parity or frozen-M03 certification. No capped allocation policy is implied.

## Gates before hosting

1. Validate actual-Gold candidate parity, cutoff/maturity, freshness, complete cohort and finite outputs. Resolve numerical sensitivity and artifact/policy compatibility.
2. Select private artifact store, batch platform and registry. Lock dependencies and use platform-managed secrets. Container/service definitions are future work.
3. Use snapshot binding + signal month + artifact digest as an idempotent run key. Publish atomically only after checks pass; fail closed on incomplete input.
4. Shadow score and monitor coverage, freshness, schema, probability drift and concentration. Evaluate calibration/returns only when labels mature.
5. Retain the approved previous artifact/binding. Rehearse rollback to exact versions and verify the restored output. Never replace provenance to bypass a failed check.
6. Seek an explicit production decision after prospective paper evidence.

These commands create no schedules, endpoints, broker connections or orders.
