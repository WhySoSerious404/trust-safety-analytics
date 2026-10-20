# Fraud detection

## Project scope

### Business question
Can we prioritize vehicle insurance claims for manual fraud review while balancing investigation workload against missed fraud?

### Dataset
The public Vehicle Insurance Fraud Detection dataset (`fraud_oracle.csv`) contains 15,420 records and 33 columns. There are 923 fraud-labelled records (5.99%). The target is `FraudFound_P`.

### Intended use
Produce a fraud ranking score and recommended review thresholds. This is a portfolio analysis using historical data; the model does not automatically reject claims or establish that fraud occurred.

### Analysis plan
1. Audit missing values, duplicates, class balance, and feature meanings.
2. Define the scoring point as initial claim review. Document feature-availability assumptions and exclusions.
3. Use a fixed 60/20/20 stratified split:

   | Set | Records | Fraud cases | Fraud rate |
   |---|---:|---:|---:|
   | Training | 9,252 | 554 | 5.99% |
   | Validation | 3,084 | 185 | 6.00% |
   | Test | 3,084 | 184 | 5.97% |

   Split membership is saved using PolicyNumber identifiers in Google Drive; the split-membership file is not included in this repository. The absence of a full claim date and customer ID limits chronological and customer-level separation.
4. Explore feature-level fraud patterns using training data only. Fit learned preprocessing within scikit-learn pipelines.
5. Compare a dummy baseline, logistic regression, and random forest, including class-weighted variants.
6. Select the model using validation average precision. Assess the selected candidate's variability through repeated stratified cross-validation within training data.
7. Run a sensitivity comparison with Fault included and review influential features against the availability audit. Fault remains excluded from the primary model regardless of the sensitivity result.
8. Choose thresholds using illustrative validation cost scenarios. Freeze final model and threshold choices before evaluating the held-out test set.

Training-only resampling, representative-grouped evaluation, and a year-based sensitivity check remain optional extensions.

### Evaluation
Prioritize precision-recall metrics because fraud is uncommon:

- Average precision and the precision-recall curve
- Precision and recall at selected thresholds
- Confusion matrix and percentage of claims flagged for review
- Fraud cases detected and missed per 1,000 claims

Validation average precision selects the model. Threshold analysis then assesses fraud detection and review workload. Accuracy is supporting context.

### Threshold selection
Compare review thresholds using:

    Total cost = (false positives x false-positive cost)
               + (false negatives x missed-fraud cost)

No real investigation costs are available. False-positive-to-missed-fraud cost ratios of 1:5, 1:10, and 1:20 are illustrative.

Compare candidate thresholds with flagging no claims or all claims. Report precision, recall, review workload, and scenario cost.

Probability calibration has not been assessed or adjusted. Thresholds will be selected by validation cost analysis rather than assuming a fixed probability cutoff such as 0.5.

These are scenario-based recommendations, not validated business optima or estimates of real savings.

## Data preparation

### Data-quality findings
- Initial checks found no pandas-recognized missing values.
- One record contained invalid "0" values in both MonthClaimed and DayOfWeekClaimed, alongside "none" in Days_Policy_Claim.
- Age was zero in 320 records (2.08%). These records exactly matched the "16 to 17" AgeOfPolicyHolder category. Other age-band mappings were also inconsistent.
- No duplicate rows were found, including after excluding PolicyNumber. All policy numbers were unique.
- All 4,987 "Sedan - Liability" records had VehicleCategory recorded as "Sport". Neither field's vehicle labels are assumed verified.
- BasePolicy repeated the coverage information in PolicyType.
- Several features had dominant categories. Rates for small groups require cautious interpretation.

### Cleaning and engineered feature
Fixed cleaning rules were applied to copies of each split:

- Convert zero Age values to missing.
- Convert "0" in MonthClaimed and DayOfWeekClaimed to missing.
- Create age_missing before converting zero ages.

The training set contained 192 zero-age records, including 23 fraud cases: an observed fraud rate of 11.98%, compared with 5.86% for nonzero ages.

The age_missing indicator captures a systematic recording pattern. It does not establish that missing age causes fraud. Ambiguous "none" values in the elapsed-day fields remain separate categories.

### Preprocessing
The primary models use **22 input columns before encoding**: 21 original features and age_missing.

- Numeric inputs: Age, Deductible, and age_missing.
- Numeric processing: median imputation followed by standard scaling.
- Categorical processing: most-frequent imputation followed by one-hot encoding.
- Categories occurring fewer than 30 times during fitting are grouped as infrequent.
- Unseen categories map to an infrequent group where available; otherwise, their encoded indicators are all zero.

The encoded feature count may differ across training folds because category grouping is learned during fitting.

Learned preprocessing uses training data only and is refitted separately within each cross-validation fold.

## Leakage checks and limitations

No accompanying data dictionary has been identified. The dataset lacks a full claim date and customer ID. Feature meanings and availability at initial review therefore involve documented assumptions.

The notebook's per-feature audit records meaning, assumed availability, modelling decision, and reasoning.

Primary model exclusions are:

- PolicyNumber: record identifier; retained for split membership.
- Fault, NumberOfSuppliments, and AddressChange_Claim: uncertain timing or meaning.
- AgeOfPolicyHolder and VehicleCategory: unresolved interpretation or consistency issues.
- BasePolicy: redundant with PolicyType.
- RepNumber: representative identifier; retained for optional grouped evaluation.
- Year: unclear event reference.
- Sex and MaritalStatus: excluded as a project scope choice.

Age and age_missing remain provisional inputs. Excluding selected demographic fields does not establish that the model is free of demographic bias.

PoliceReportFiled, WitnessPresent, AgentType, and DriverRating are included under assumptions that their values were available at initial review rather than updated after investigation.

Duplicate checks cannot establish customer-level separation. Grouping by RepNumber would assess representative-level separation only.

Year ranges from 1994 to 1996, with 6,142, 5,195, and 4,083 records respectively. Its meaning must be established before interpreting a year-based split as chronological evaluation.

Feature importance can identify fields requiring closer review, but cannot establish whether leakage is present or absent. Availability decisions require reasoning about field meaning and timing.

The primary stratified evaluation does not establish performance on future claims.

## Initial model results

Weighted random forest achieved the highest validation average precision. Its operational usefulness remains to be assessed through threshold analysis.

| Model | Validation average precision |
|---|---:|
| Dummy baseline | 0.0600 |
| Logistic regression | 0.1142 |
| Weighted logistic regression | 0.1097 |
| Random forest | 0.1426 |
| Weighted random forest | 0.1462 |

The constant-score dummy's AP equals the validation fraud rate. This is a no-ranking reference, not a performance floor.

Weighted random forest achieved approximately 2.4 times the reference AP. This does not mean 2.4 times as much fraud would be caught at a fixed review workload.

Class weighting reduced logistic regression AP and slightly improved random forest AP. Weighted random forest is carried forward because it achieved the highest validation AP; its small advantage has not been established as consistent.

### Model settings
- Dummy: strategy="prior".
- Logistic regression: max_iter=1000; compared default class weighting with class_weight="balanced".
- Random forest: n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1; compared default class weighting with class_weight="balanced".
- One-hot encoder: min_frequency=30, handle_unknown="infrequent_if_exist".
- Other model settings were left at library defaults.

These were predefined starting configurations. No hyperparameter search was performed.

### Stability check
Cross-validation was run for **weighted random forest only**, using five stratified folds repeated twice within training data.

| Model | Mean CV AP | Standard deviation | Minimum AP | Maximum AP |
|---|---:|---:|---:|---:|
| Weighted random forest | 0.1342 | 0.0227 | 0.1096 | 0.1889 |

Preprocessing was fitted separately within each fold.

The variation reflects changes in the training sample and fold assignment. Repeated folds are not independent; the standard deviation is descriptive, not a confidence interval.

The other models were compared on a single validation split. No paired cross-validation comparison between weighted and unweighted forests was performed.

### Remaining evaluation
- Precision-recall curve for the selected model against the dummy reference
- Cost-based threshold selection using validation scores
- Fault sensitivity comparison
- Review of influential features against uncertain availability assumptions
- Final model and threshold freeze
- One held-out test evaluation

The test set has not been used for model-performance evaluation.

## Deliverables
- Reproducible Colab notebook
- Data-quality findings and per-feature leakage audit
- Model comparison and precision-recall plots
- Training-only stability summary
- Cost-scenario and threshold results
- Final README results section
