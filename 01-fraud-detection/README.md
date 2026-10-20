# Fraud detection

## Project scope

### Business question
Can we prioritize vehicle insurance claims for manual fraud review while balancing investigation workload against missed fraud?

### Dataset
The public Vehicle Insurance Fraud Detection dataset (`fraud_oracle.csv`) contains 15,420 records and 33 columns. There are 923 fraud-labelled records (5.99%). The target is `FraudFound_P`.

### Intended use
Produce a fraud risk score and recommended review thresholds. This is a portfolio analysis using historical data; the model does not automatically reject claims or establish that fraud occurred.

### Analysis plan
1. Audit missing values, duplicates, class balance, and feature meanings.
2. Define the scoring point as initial claim review. Document feature-availability assumptions and exclusions.
3. Use a fixed 60/20/20 stratified split:

   | Set | Records | Fraud cases | Fraud rate |
   |---|---:|---:|---:|
   | Training | 9,252 | 554 | 5.99% |
   | Validation | 3,084 | 185 | 6.00% |
   | Test | 3,084 | 184 | 5.97% |

   Split membership is saved using PolicyNumber identifiers. The absence of a full claim date and customer ID limits chronological and customer-level separation.
4. Explore feature-level fraud patterns using training data only. Fit learned preprocessing within scikit-learn pipelines.
5. Compare a dummy baseline, logistic regression, and random forest, including class-weighted variants. One training-only resampling method remains optional.
6. Select the model using validation average precision. Assess its variability through repeated stratified cross-validation within training data.
7. Compare the selected model with a version including Fault as a sensitivity check, using validation predictions.
8. Choose thresholds using illustrative cost scenarios on validation predictions. Freeze model and threshold choices before final held-out test evaluation.

### Evaluation
Prioritize precision-recall metrics because fraud is uncommon:

- Average precision and the precision-recall curve
- Precision and recall at selected thresholds
- Confusion matrix and percentage of claims flagged for review
- Fraud cases detected and missed per 1,000 claims

Validation average precision selects the model. Threshold analysis then assesses the trade-off between fraud detection and review workload. Accuracy is supporting context.

### Threshold selection
Compare review thresholds using:

    Total cost = (false positives x false-positive cost)
               + (false negatives x missed-fraud cost)

No real investigation costs are available. False-positive-to-missed-fraud cost ratios of 1:5, 1:10, and 1:20 are illustrative.

Report how threshold choices affect precision, recall, review workload, and scenario cost. These are scenario-based recommendations, not validated business optima or estimates of real savings.

## Data preparation

### Data-quality findings
- No pandas-recognized missing values were initially found, but one record contained invalid "0" values in both MonthClaimed and DayOfWeekClaimed.
- Age was zero in 320 records (2.08%). These records exactly matched the "16 to 17" AgeOfPolicyHolder category. Other age-band mappings were also inconsistent.
- No duplicate rows were found, including after excluding PolicyNumber. All policy numbers were unique.
- All 4,987 "Sedan - Liability" records had VehicleCategory recorded as "Sport". Neither field's vehicle labels are assumed verified.
- BasePolicy repeated the coverage information in PolicyType.
- Several categorical features were dominated by one value; rare-group rates require cautious interpretation.

### Cleaning and preprocessing
Fixed cleaning rules were applied to copies of each split:

- Convert zero Age values to missing.
- Convert "0" in MonthClaimed and DayOfWeekClaimed to missing.
- Create age_missing before converting zero ages.

The training set contained 192 zero-age records, including 23 fraud cases: an observed fraud rate of 11.98%, compared with 5.86% for nonzero ages. The indicator captures a systematic recording pattern, not evidence that missing age causes fraud.

The primary models use 22 inputs: 21 original features and age_missing.

- Numeric inputs: median imputation followed by standard scaling.
- Categorical inputs: most-frequent imputation followed by one-hot encoding.
- Categories occurring fewer than 30 times during fitting are grouped as infrequent.
- Unseen categories map to the infrequent group where available; otherwise, their encoded indicators are all zero.

Preprocessing is learned from training data only and refitted separately within each cross-validation fold.

## Leakage checks and limitations

The available dataset has no full claim date, customer ID, or accompanying data dictionary. Feature meanings and availability at initial review therefore involve documented assumptions.

The per-feature audit in the notebook records each column's assumed meaning, availability, modelling decision, and reasoning.

Primary model exclusions are:

- PolicyNumber: record identifier; retained for split membership.
- Fault, NumberOfSuppliments, and AddressChange_Claim: uncertain timing or meaning.
- AgeOfPolicyHolder and VehicleCategory: unresolved interpretation or consistency issues.
- BasePolicy: redundant with PolicyType.
- RepNumber: representative identifier; retained for optional grouped evaluation.
- Year: unclear event reference.
- Sex and MaritalStatus: excluded as a project scope choice.

Age and age_missing remain provisional inputs. Excluding selected demographic fields does not establish that the model is free of demographic bias.

PoliceReportFiled, WitnessPresent, AgentType, and DriverRating are included under explicit assumptions that their recorded values were available at initial review rather than updated after investigation.

Duplicate checks cannot establish customer-level separation. Optional grouping by RepNumber would assess representative-level separation only.

Year ranges from 1994 to 1996, with 6,142, 5,195, and 4,083 records respectively. Its meaning must be established before interpreting a year-based split as chronological evaluation. The primary stratified evaluation does not establish performance on future claims.

## Initial model results

| Model | Validation average precision |
|---|---:|
| Dummy baseline | 0.0600 |
| Logistic regression | 0.1142 |
| Weighted logistic regression | 0.1097 |
| Random forest | 0.1426 |
| Weighted random forest | 0.1462 |

Weighted random forest achieved the highest validation AP and was provisionally selected for threshold analysis.

Both forests used 300 trees and a minimum of five training records per leaf. These were starting settings, not extensively tuned optima.

Class weighting reduced logistic regression AP and slightly improved random forest AP. The small forest improvement has not been established as consistent across samples.

### Stability check
Training-only five-fold stratified cross-validation, repeated twice, produced these weighted-random-forest results:

- Mean AP: 0.1342
- Standard deviation: 0.0227
- Range: 0.1096–0.1889

The spread indicates sensitivity to sample composition. Repeated folds are not independent, and the standard deviation is descriptive rather than a confidence interval.

AP summarizes ranking quality; it does not specify precision or recall at one review threshold.

### Remaining work
- Fault sensitivity comparison
- Cost-based threshold analysis
- Final held-out test evaluation
- Stakeholder recommendations

The test set has not been used for model-performance evaluation.

## Deliverables
- Reproducible Colab notebook
- Data-quality findings and per-feature leakage audit
- Model comparison and precision-recall plots
- Cross-validation stability summary
- Cost-scenario and threshold results
- Final README results section
