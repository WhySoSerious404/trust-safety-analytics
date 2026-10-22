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

Weighted random forest achieved the highest validation average precision and was selected for the frozen threshold analysis reported below.

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

## Validation threshold analysis

Thresholds were selected for the weighted random forest by minimizing:

    Scenario cost = FP + c x FN

The search covered every distinct validation score, plus a flag-nothing option. FP cost is set to 1; missed-fraud cost c is 5, 10 or 20. Equal-cost choices favour lower review workload.

| FP:FN cost ratio | Threshold* | Precision | Recall | Review share | Scenario cost |
|---|---:|---:|---:|---:|---:|
| 1:5 | 0.5095 | 35.90% | 7.57% | 1.26% | 880 |
| 1:10 | 0.3305 | 13.07% | 42.16% | 19.36% | 1,589 |
| 1:20 | 0.1721 | 8.95% | 94.05% | 63.04% | 1,990 |

*Displayed thresholds are rounded; full-precision values are retained in `scenario_results`.

Each choice had lower validation scenario cost than flagging nothing or everything, though at 1:5 the margin over flagging nothing is small (880 vs 925). These comparisons use the same data that selected the thresholds, so they require held-out evaluation.

The three ratios represent different operating points:

- At 1:5, the threshold flags very few claims, so its result rests on a small number of true positives.
- At 1:10, reviewing about 19% of claims catches about 42% of fraud at 13% precision.
- At 1:20, reviewing about 63% of claims catches about 94% of fraud at 8.95% precision, requiring substantial review capacity. Approximately 37% of claims remain unreviewed.

The benefit of the model depends heavily on the assumed cost of missed fraud. The 1:10 scenario is carried forward as the primary illustrative scenario, with 1:5 and 1:20 as alternatives. This is not a validated business recommendation.

Costs are hypothetical penalties, not actual financial losses. The formula omits investigation costs for true-positive claims, so review workload is reported separately.

## Fault sensitivity and importance review

Adding Fault increased validation AP from 0.1462 to 0.2097 on a single validation split containing 185 fraud cases. This shows Fault carries predictive information, not that it is available at initial review. Fault remains excluded from the primary model regardless of this result.

The primary forest's leading individual encoded features included the Sedan-Liability policy indicator and Age, with DriverRating categories also among the leaders. Sedan-Liability belongs to the low-fraud Liability group and is affected by the vehicle-label discrepancy documented earlier.

Age is itself a demographic input. Group-level disparities and relationships with excluded demographic fields have not been assessed. DriverRating's availability at initial review is assumed, not known.

Importance values describe training split improvements. They do not establish direction, causation or absence of leakage. Numeric-feature bias, correlated inputs, uncertain field meanings and availability assumptions remain limitations.

The sensitivity comparison and importance review did not use test performance to change model choices.

## Frozen model and threshold choices

These choices are fixed before the first held-out test evaluation.

### Primary model
Use the existing weighted random forest pipeline fitted on training data only: `rf_balanced_pipeline`.

Model settings:
- n_estimators=300
- min_samples_leaf=5
- class_weight="balanced"
- random_state=42
- n_jobs=-1
- Other forest settings remain at library defaults.

Retain the existing cleaning and preprocessing:
- Zero ages and invalid claim-calendar placeholders become missing.
- age_missing preserves the original zero-age indicator.
- Numeric inputs use median imputation and standard scaling.
- Categorical inputs use most-frequent imputation and one-hot encoding.
- Encoder settings: min_frequency=30 and
  handle_unknown="infrequent_if_exist".

All learned preprocessing was fitted on training data.

### Frozen inputs
The 22 input columns before encoding are:

Month, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed, Age, PolicyType, VehiclePrice, Deductible, DriverRating, Days_Policy_Accident, Days_Policy_Claim, PastNumberOfClaims, AgeOfVehicle, PoliceReportFiled, WitnessPresent, AgentType, NumberOfCars, age_missing.

Fault remains excluded. Its sensitivity model will not replace the primary model.

### Frozen thresholds

Flag a claim when its primary model score is greater than or equal to the scenario threshold.

| FP:FN cost ratio | Full-precision threshold | Role |
|---|---:|---|
| 1:5 | 0.50949964720280316 | Alternative |
| 1:10 | 0.33054037832098099 | Primary illustrative scenario |
| 1:20 | 0.17208923258978076 | Alternative |

The 1:10 scenario is an illustrative reporting choice, not a validated business recommendation.

### Saved model

The fitted pipeline, thresholds, input-column list, and primary scenario are saved in Google Drive as frozen_fraud_model.joblib. This artifact is not included in the repository.

Test evaluation will use this saved fitted pipeline and frozen thresholds, with no further fitting or threshold adjustment. No test-set performance has been examined.

### Held-out evaluation plan
Evaluate the existing fitted primary pipeline on the test set once. Do not refit on training plus validation.

Report test average precision once, since all three scenarios use the same model scores.

For each frozen threshold, report:
- Confusion matrix
- Precision and recall
- Number and percentage of claims flagged
- Fraud detected and missed per 1,000 claims
- Scenario cost and flag-nothing/flag-everything comparisons


## Held-out test results

At the primary illustrative 1:10 operating point, reviewing 19.55% of test claims caught 39.67% of fraud, at 12.11% precision.

The test set contained 3,084 claims, including 184 fraud cases.
Evaluation used the saved fitted model and frozen thresholds, without refitting or adjusting choices after seeing test results.

### Ranking performance

| Metric | Validation | Test |
|---|---:|---:|
| Average precision | 0.1462 | 0.1247 |

Ranking performance was weaker on the held-out test set.

Sampling variation may contribute to the AP difference. However, the training cross-validation SD of 0.023 does not establish whether the validation-to-test drop is attributable to sampling noise.

### Frozen operating points

Detected and missed counts below are standardized per 1,000 total claims, not per 1,000 fraud cases.

| Cost ratio | Precision | Recall | Review share | Fraud detected per 1,000 | Fraud missed per 1,000 |
|---|---:|---:|---:|---:|---:|
| 1:5 | 19.51% | 4.35% | 1.33% | 2.59 | 57.07 |
| 1:10 | 12.11% | 39.67% | 19.55% | 23.67 | 35.99 |
| 1:20 | 8.44% | 88.59% | 62.61% | 52.85 | 6.81 |

### Confusion-matrix counts

Positive means flagged for fraud review.

| Cost ratio | True positives | False positives | False negatives | True negatives |
|---|---:|---:|---:|---:|
| 1:5 | 8 | 33 | 176 | 2,867 |
| 1:10 | 73 | 530 | 111 | 2,370 |
| 1:20 | 163 | 1,768 | 21 | 1,132 |

### Illustrative scenario costs

| Cost ratio | Model cost | Flag nothing | Flag everything |
|---|---:|---:|---:|
| 1:5 | 913 | 920 | 2,900 |
| 1:10 | 1,640 | 1,840 | 2,900 |
| 1:20 | 2,188 | 3,680 | 2,900 |

Flag nothing = missed-fraud cost x 184 fraud cases. Flag everything = 2,900 non-fraud claims at FP cost 1.

All three frozen operating points had lower observed test scenario cost than either simple alternative.

At 1:10, cost was 200 units (10.87%) below flagging nothing, the better simple alternative. At 1:5, the improvement was only seven units and rested on eight detected fraud cases.

Costs are hypothetical penalties, not actual financial losses.
Investigation costs for true-positive claims are omitted; review workload is reported separately.

### Interpretation

The model provides a trade-off between review workload and fraud detection. The primary illustrative scenario catches approximately 40% of fraud while flagging approximately 20% of claims, but most flagged claims are non-fraud.

Higher recall requires substantially more review capacity.
Real operating costs and capacity constraints would be needed to choose a business threshold.

These findings do not establish future performance, production readiness, or grounds for automated claim denial.

## Deliverables
- Reproducible Colab notebook: 01-fraud-detection/fraud_analysis.ipynb
- Data-quality findings and per-feature leakage audit
- Model comparison and precision-recall plots
- Training-only stability summary
- Cost-scenario and threshold results
- Final README results section
