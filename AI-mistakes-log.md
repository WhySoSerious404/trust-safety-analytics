# AI mistakes log

Significant cases where an AI tool gave wrong, incomplete, or unverified output that could have affected results, and how it was caught. Minor slips (typos, syntax errors, formatting) are not logged.

## Log

| # | Tool | What went wrong | How it was caught | Fix |
|---|------|-----------------|-------------------|-----|
| 1 | Claude | Scope did not say which data the cost-based threshold analysis used. Run on the test set, this would tune on held-out data (leakage). | Codex reviewed the scope and flagged it. | Thresholds are now chosen on validation predictions, then evaluated once on the test set. |
