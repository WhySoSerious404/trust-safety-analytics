# AI mistakes log

Significant cases where an AI tool gave wrong, incomplete, or unverified output that could have affected results, and how it was caught. Minor slips (typos, syntax errors, formatting) are not logged.

## Log

| # | Tool | What went wrong | How it was caught | Fix |
|---|------|-----------------|-------------------|-----|
| 1 | Claude | Scope did not say which data the cost-based threshold analysis used. Run on the test set, this would tune on held-out data (leakage). | Codex reviewed the scope and flagged it. | Thresholds are chosen on validation predictions, then evaluated once on the test set. |
| 2 | Claude | Described the dummy baseline's AP as equal to the training fraud rate and as a "floor", and called the two random forests "indistinguishable" using a CV spread measured on one model only. Also suggested class weighting "may stabilise ranking" with no evidence. | Codex pointed out each claim went beyond the evidence. | Now: dummy AP equals the validation fraud rate and is a "reference", not a floor; the forest difference is "not established as consistent"; the stabilisation claim was dropped. |
