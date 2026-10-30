# Stakeholder memo: prioritising fraud and abuse review

## Summary

The claims model is a modest ranking aid: it finds fraud-labelled claims at about twice the base rate, but most flagged claims are not fraud. The platform-abuse rules perform well on planted patterns, but they are demonstrations, not evidence of real-world performance. Neither analysis establishes wrongdoing or supports automatic rejection or enforcement.

## Insurance claims

The dataset contained 15,420 vehicle insurance claims, 923 of them fraud-labelled (5.99%). A weighted random forest was selected using validation average precision, a summary of how well a model ranks fraud-labelled claims above the rest. The model and thresholds were fixed before the held-out test set was evaluated.

Test average precision was 0.1247, against 0.1462 on validation. At the primary illustrative cost ratio of 1:10 (one unit for a wrongly flagged claim, ten for a missed fraud), the frozen threshold:

- Flagged 603 of 3,084 test claims for review (19.55%).
- Caught 73 of 184 fraud-labelled claims (39.67% recall).
- Achieved 12.11% precision against a 5.97% test fraud rate, so about twice the base rate. 530 flagged claims were not fraud-labelled.
- Missed 111 fraud-labelled claims.

The 1:10 ratio is an assumption, and the result depends on it. At 1:5 the model flagged 41 claims and caught only 8 fraud cases; at 1:20 it caught 88.59% of fraud but required reviewing 62.61% of claims.

At 1:10 the scenario penalty was 1,640 units, against 1,840 for reviewing nothing and 2,900 for reviewing everything. These are hypothetical penalties, not measured savings, and they exclude the cost of investigating true positives.

## Platform abuse

SQL rules were developed on 20,000 synthetic accounts and 300,000 events. For each pattern, the selected rule favoured recall while considering precision and review workload.

| Pattern | Selected rule | Flagged | Precision | Recall | Legitimate false alarms |
|---|---|---:|---:|---:|---:|
| Automated clicking | 5 clicks within 60 seconds | 431 | 59.63% | 85.67% | 174 |
| Spam | 15 accounts posting from one /24 block within 120 seconds | 483 | 82.82% | 100% | 83 |
| Takeover | Login from an unfamiliar country, then an hourly activity gain of at least 4 | 348 | 85.63% | 99.33% | 50 |

Review shares were 2.155%, 2.415% and 1.740% of accounts, each measured separately.

Every legitimate false alarm came from the simulated lookalike group built for that pattern: genuine click spikes, shared-network activity and travel. Suspicious behaviour therefore overlaps with legitimate use, and flags need human review.

The takeover rule requires at least five earlier events and compares activity in the following hour with the preceding hour. The implemented rule is evaluated one hour after login, when the full following-hour count is available.

## Recommendation and limits

Treat both analyses as demonstrations of review prioritisation and trade-off measurement, not production-ready decisions.

**Claims.** Feature availability at claim review could not be verified. Customer-level separation and reliable chronological evaluation were not possible. Demographic disparities were not assessed.

**Platform abuse.** Thresholds were developed and selected on the same synthetic dataset, informed by evaluation labels. High recall reflects how the patterns were simulated and does not establish real-world effectiveness. The takeover baseline is particularly sparse; a real rule would compare against a longer history.

**Before operational use:**
- Obtain representative data and reliable labels.
- Verify feature availability and alert timing.
- Evaluate on later, unseen activity.
- Assess outcomes across subgroups.
- Establish review capacity and actual investigation costs.
- Ensure human review precedes any adverse action.
