# Fraud detection

## Project scope

### Business question
Can we prioritize vehicle insurance claims for manual fraud review while balancing investigation workload against missed fraud?

### Dataset
Use the public Vehicle Insurance Fraud Detection dataset (`fraud_oracle.csv`), containing 15,420 records and 33 columns, with approximately 6% fraud, to be confirmed during exploration. The target is `FraudFound_P`.

### Intended use
Produce a fraud risk score and a recommended review threshold. This is a portfolio analysis using historical data; the model does not automatically reject claims or establish that fraud occurred.

### Analysis plan
1. Audit missing values, duplicates, class balance, and feature meanings.
2. Define the scoring point as initial claim review. Document assumptions about each feature's availability and exclude fields judged unavailable at that point.
3. Create a 60/20/20 stratified training, validation, and held-out test split (about 9,250 / 3,085 / 3,085 claims, with roughly 185 fraud cases in each of validation and test) after checking duplicates. The dataset lacks a full claim date and customer ID, so chronological and customer-level separation cannot be reliably established. Document these limitations; optionally assess grouping by `RepNumber`.
4. Fit preprocessing only on training data through scikit-learn pipelines. Exclude `PolicyNumber`, which functions as a row identifier.
5. Compare a dummy baseline, logistic regression, and a random forest.
6. Compare class weighting and, if time allows, one training-only resampling method. Apply resampling only to the training data (inside cross-validation folds if CV is used), leaving validation and test distributions unchanged.
7. Select the model using validation average precision. Choose a threshold for each illustrative cost ratio using validation predictions, then evaluate the selected model and thresholds once on the held-out test set.
8. Assess metric variability within the training data using repeated stratified cross-validation or several random seeds, keeping the original validation and test sets fixed. Fraud counts in the held-out sets are small, so single-split metrics are noisy.

### Evaluation
Prioritize precision-recall metrics because fraud is uncommon:
- Average precision and the precision-recall curve
- Precision and recall at the selected threshold
- Confusion matrix and percentage of claims flagged for review
- Fraud cases detected and missed per 1,000 claims

Average precision on the validation set selects the model; thresholds are then chosen from the cost analysis on validation predictions. Accuracy is reported only as supporting context.

### Threshold selection
Compare review thresholds using:

    Total cost = (false positives x false-positive cost)
               + (false negatives x missed-fraud cost)

No real investigation costs are available, so false-positive-to-missed-fraud cost ratios such as 1:5, 1:10, and 1:20 are illustrative.

Show how the selected threshold, fraud recall, and review workload change across these scenarios. The result is a scenario-based recommendation, not a validated business optimum.

### Leakage checks and limitations
The dataset has no full claim date, customer ID, or accompanying data dictionary. Feature availability at scoring time is therefore an assumption to document per feature.

`PoliceReportFiled`, `WitnessPresent`, and `AgentType` are treated as plausibly available at filing time, subject to an explicitly documented assumption.

Repeated-entity checks are limited to duplicate rows and optional grouping by `RepNumber`, an agent-style identifier; these cannot establish customer-level separation. `PolicyNumber` functions as a row index.

A split by `Year` could provide a coarse temporal sensitivity check, but may produce small or imbalanced subsets. The primary stratified evaluation does not establish performance on future claims.

Review target-related fields, potentially post-investigation information, duplicates, and preprocessing performed before splitting. Document exclusions and unresolved risks.

### Deliverables
- Reproducible Colab notebook with exploration, preprocessing, imbalance handling, model comparison, and threshold analysis
- Results table and precision-recall plots
- Documented leakage audit and limitations
- A per-feature table (column, meaning, assumed available at scoring time, reasoning)
- README results section
