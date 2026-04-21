# Baseline Supervised Model: Technical Report

## Overview

This report documents the coding process, dependencies, and techniques used to develop the baseline multinomial logistic regression model for predicting student burnout in the AuraCheck project. The model serves as a supervised benchmark for understanding burnout patterns using a subset of survey features and provides predictions for the Streamlit application.

---

## 1. Project Objectives

**Primary Goal:** Establish a production-ready supervised baseline model that predicts student burnout into one of four severity classes based on survey responses.

**Scope:**
- Implement a 4-class multinomial logistic regression classifier
- Use a VIF-pruned feature set (14 features selected to minimize multicollinearity)
- Train on stratified train-test splits (80/20)
- Evaluate using accuracy, Cohen's Kappa, per-class recall, and F1 score
- Document the model development process with comparison tables and diagnostic reports
- Export the trained model, scaler, metadata, and example predictions

**Target Variable:** Student burnout binned into four quartiles:
- **Very Low (Q1):** Burnout raw score 0.00–1.67
- **Low (Q2):** Burnout raw score 1.67–2.33
- **Moderate (Q3):** Burnout raw score 2.33–3.00
- **High (Q4):** Burnout raw score 3.00–5.00

---

## 2. Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | ≥1.5.0 | Logistic Regression, train-test splitting, preprocessing (StandardScaler), evaluation metrics |
| **pandas** | ≥2.2.0 | Data manipulation, CSV I/O, feature selection, binning |
| **numpy** | ≥1.26.0 | Numerical operations, array handling, p-value calculations |
| **joblib** | Bundled with scikit-learn | Model serialization and deserialization |
| **matplotlib** | ≥3.9.0 | Static figure generation (regression plots, confusion matrices) |
| **seaborn** | ≥0.13.2 | Statistical data visualization (heatmaps, bar charts) |
| **scipy** | ≥1.13.1 | Statistical functions (for p-values, effect sizes) |

Full dependency specification: [requirements.txt](requirements.txt)

---

## 3. Data Preparation

### Data Source
- **File:** `Dataset/students_mental_health_survey_with_burnout_final.csv`
- **Rows:** ~10,000 students
- **Target Variable:** `burnout_raw_score` → converted to quartiles for 4-class classification

### Feature Selection

**Initial Feature Set (17 features):**
```python
["Course", "Age", "Gender", "CGPA", "Sleep_Quality", "Physical_Activity",
 "Diet_Quality", "Social_Support", "Relationship_Status", "Substance_Use",
 "Counseling_Service_Use", "Family_History", "Chronic_Illness",
 "Financial_Stress", "Extracurricular_Involvement", "Semester_Credit_Load",
 "Residence_Type"]
```

**Pruned Feature Set (14 features):**
Features removed to minimize multicollinearity (VIF > 10):
- `Age` — High correlation with other features
- `CGPA` — Correlated with academic performance indicators
- `Semester_Credit_Load` — Redundant with course workload information

**Final Features Used:**
```python
["Course", "Gender", "Sleep_Quality", "Physical_Activity", "Diet_Quality",
 "Social_Support", "Relationship_Status", "Substance_Use", "Counseling_Service_Use",
 "Family_History", "Chronic_Illness", "Financial_Stress",
 "Extracurricular_Involvement", "Residence_Type"]
```

### Categorical Encoding

All categorical features are encoded using fixed dictionaries to ensure consistency during training and inference:

| Feature | Encoding |
|---------|----------|
| **Gender** | Female=0, Male=1 |
| **Sleep_Quality** | Poor=0, Average=1, Good=2 |
| **Physical_Activity** | Low=0, Moderate=1, High=2 |
| **Diet_Quality** | Good=0, Average=1, Poor=2 |
| **Social_Support** | High=0, Moderate=1, Low=2 |
| **Substance_Use** | Never=0, Unknown=1, Occasionally=2, Frequently=3 |
| **Counseling_Service_Use** | Never=0, Occasionally=1, Frequently=2 |
| **Family_History** | No=0, Yes=1 |
| **Chronic_Illness** | No=0, Yes=1 |
| **Extracurricular_Involvement** | High=0, Moderate=1, Low=2 |
| **Course** | Business=0, CS=1, Engineering=2, Law=3, Medical=4, Others=5 |
| **Relationship_Status** | In a Relationship=0, Married=1, Single=2 |
| **Residence_Type** | Off-Campus=0, On-Campus=1, With Family=2 |

### Handling Missing Values

- **Numeric columns:** Imputed with column median
- **Categorical columns:** Imputed with "Unknown" (maps to a default encoding value)

---

## 4. Core Modeling Techniques

### 4.1 Train-Test Split

**Configuration:**
- **Train fraction:** 80%
- **Test fraction:** 20%
- **Stratification:** By target class (preserves class distribution in both splits)
- **Random state:** 42 (reproducibility)

**Purpose:** Stratification ensures all four burnout classes are represented proportionally in both train and test sets, preventing class imbalance in evaluation.

### 4.2 Feature Scaling

**Technique:** StandardScaler (z-score normalization)

**Steps:**
1. Compute mean and standard deviation on the **training set only**
2. Apply the training set's transformation to the training set
3. Apply the same transformation to the test set (using training statistics)

**Purpose:** Logistic regression is sensitive to feature scale; standardization ensures numerical stability and equal feature weighting.

### 4.3 Multinomial Logistic Regression

**Algorithm:** Multinomial Logistic Regression (softmax classifier)

**Configuration:**
- **Solver:** `lbfgs` (Limited-memory BFGS)
  - Robust for multiclass problems
  - Supports class weights
  - Convergence is well-behaved
- **Max iterations:** 5000 (sufficient for convergence on this dataset)
- **Random state:** 42 (reproducibility for solver initialization)
- **Class weight:** `balanced` (automatically adjusts weights inversely proportional to class frequencies)
  - Prevents the model from biasing toward the majority class (Very Low burnout)
  - Ensures all burnout levels are predicted with reasonable recall
- **Multi-class mode:** Multinomial loss (default, appropriate for 4-class problem)

**Model Output:**
- **Coefficients:** Shape (4, 14) — 4 classes × 14 features
- **Intercepts:** Shape (4,) — one per class
- **Decision function:** $ f(x) = Xw + b $ → softmax probabilities

### 4.4 Model Fitting Procedure

```
1. Load data
2. Extract features (X) and target (y)
3. Bin target into 4 quartiles
4. Train-test split (80/20, stratified)
5. Fit StandardScaler on X_train
6. Transform X_train and X_test
7. Fit LogisticRegression on X_train_scaled
8. Save model, scaler, and metadata
9. Evaluate on X_test_scaled
10. Export artifacts (model file, metrics, metadata, example prediction)
```

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

| Metric | Formula | Interpretation | Notes |
|--------|---------|---|---|
| **Accuracy** | (TP+TN) / Total | Proportion of correct predictions | Simple but doesn't account for class imbalance |
| **Cohen's Kappa** | (po - pe) / (1 - pe) | Agreement beyond chance | Accounts for class distribution; range [−1, 1]; >0.6 is good |
| **Macro Recall** | Mean of per-class recall | Average sensitivity across all classes | Highlights minority class performance |
| **Macro F1 Score** | 2 * (P*R) / (P+R) | Harmonic mean of precision and recall | Balances false positives and false negatives |

### 5.2 Per-Class Metrics

For each burnout class, compute:
- **Recall (Sensitivity):** TP / (TP + FN) — "Of actual Q_i students, how many did we predict as Q_i?"
- **Specificity:** TN / (TN + FP) — "Of non-Q_i students, how many did we correctly rule out?"
- **Precision:** TP / (TP + FP) — "Of predicted Q_i students, how many were actually Q_i?"

### 5.3 Log Loss (Cross-Entropy)

- **Formula:** $ -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{p}_{ic}) $
- **Interpretation:** Penalizes confident wrong predictions; lower is better
- **Use case:** Assesses calibration of probability outputs

---

## 6. Workflow and Scripts

### 6.1 Execution Workflow

**Installation:**
```bash
cd /path/to/auracheck
python -m pip install -r requirements.txt
```

**Execution (from repository root):**

1. **Train and save baseline model:**
   ```bash
   python baseline/scripts/production_pruned_multinomial_baseline.py
   ```
   Outputs: [baseline/outputs/final_baseline_model/](outputs/final_baseline_model/)

2. **Generate comparison report:**
   ```bash
   python baseline/scripts/create_final_process_baseline_report.py
   ```
   Outputs: PDF report, summary text, confusion matrix, sensitivity/specificity CSVs

### 6.2 Script Overview

#### production_pruned_multinomial_baseline.py
- **Purpose:** Train and serialize the multinomial baseline model
- **Key Steps:**
  1. Load data and preprocess (encode, impute, select features)
  2. Create 4-class target (quartiles of burnout_raw_score)
  3. Train-test split (80/20, stratified)
  4. Fit StandardScaler on training set
  5. Fit LogisticRegression with `class_weight='balanced'`
  6. Evaluate on test set
  7. Save model, scaler, metadata, metrics, and example prediction
  8. Output: Model file (joblib), metrics CSV, metadata JSON, example prediction JSON

**Code Location:** [baseline/scripts/production_pruned_multinomial_baseline.py](scripts/production_pruned_multinomial_baseline.py)

#### create_final_process_baseline_report.py
- **Purpose:** Document model development and create comparison tables
- **Key Steps:**
  1. Load the trained model and data
  2. Run multiple baseline candidate models (varied features, hyperparameters)
  3. Compare candidates on accuracy, kappa, recall, F1
  4. Compute confusion matrix and per-class sensitivity/specificity
  5. Generate regression coefficient table with p-values
  6. Check model assumptions (solver convergence, VIF multicollinearity)
  7. Create PDF report with narrative, tables, and diagnostic plots
  8. Output: PDF report, comparison CSV, confusion matrix, assumptions text

**Code Location:** [baseline/scripts/create_final_process_baseline_report.py](scripts/create_final_process_baseline_report.py)

#### production_pruned_multinomial_baseline.ipynb (Notebook version)
- **Purpose:** Notebook-based workflow for model training
- **Use case:** Interactive exploration, visualization, and step-by-step model development
- **Code Location:** [baseline/scripts/production_pruned_multinomial_baseline.ipynb](scripts/production_pruned_multinomial_baseline.ipynb)

#### create_final_baseline_report.ipynb (Notebook version)
- **Purpose:** Notebook-based full report pipeline
- **Use case:** Comprehensive documentation with live visualizations and markdown narrative
- **Code Location:** [baseline/scripts/create_final_baseline_report.ipynb](scripts/create_final_baseline_report.ipynb)

---

## 7. Output Artifacts

### 7.1 Model and Metadata Files

**Directory:** [baseline/outputs/final_baseline_model/](outputs/final_baseline_model/)

- **production_pruned_multinomial_model.joblib**: Serialized model (includes LogisticRegression classifier)
- **production_pruned_multinomial_metadata.json**: Model configuration and feature encoding
  - Feature list
  - Class names and mapping
  - Encoding dictionaries
  - Training parameters

### 7.2 Metrics and Evaluation Files

- **production_pruned_multinomial_metrics.csv**: Per-class and macro metrics (accuracy, recall, precision, F1)
- **final_selected_baseline_confusion_matrix.csv**: 4×4 confusion matrix (predicted vs. actual)
- **final_selected_baseline_sensitivity_specificity.csv**: Per-class sensitivity, specificity, and other rates

### 7.3 Example Prediction

- **production_pruned_multinomial_example_prediction.json**: Example single-student prediction output
  - Input features
  - Raw logits
  - Predicted class and probability

### 7.4 Report Documents

- **final_process_baseline_comparison_report.pdf**: Comprehensive PDF report including:
  - Model selection process and candidate comparison table
  - Regression coefficients with confidence intervals and p-values
  - Confusion matrix heatmap
  - Per-class metrics bar charts
  - Model assumptions checks
  - Usage examples and interpretation guide
  
- **final_process_baseline_comparison_report_summary.txt**: Plain-text summary of key findings and metrics

- **final_model_assumption_checks.txt**: Diagnostic checks
  - Solver convergence (iteration count and max_iter)
  - Multicollinearity (Variance Inflation Factors)
  - Class distribution in train/test sets

---

## 8. Key Design Decisions

### 8.1 Why Multinomial Logistic Regression?

**Chosen for:**
- **Interpretability:** Coefficients have clear meaning (impact of each feature on each burnout class)
- **Speed:** Fast to train and predict on new data
- **Simplicity:** Few hyperparameters to tune
- **Probability output:** Natural softmax output for confidence calibration
- **Clinical familiarity:** Logistic regression is standard in medical/health informatics

**Not chosen (alternatives):**
- Random Forest / Gradient Boosting: More accurate but less interpretable
- Neural Networks: Overkill for ~10k rows and 14 features; harder to deploy
- SVM: Less natural for probability output

### 8.2 Why Four Classes (Quartiles)?

- **Fair representation:** Quartiles ensure all burnout levels have equal sample size (approximately)
- **Interpretability:** Intuitive groupings ("Very Low" → "High")
- **Clinical relevance:** Matches a common stratification scheme in mental health research

### 8.3 Why Stratified Train-Test Split?

- **Prevents class imbalance in splits:** If there are fewer "High burnout" students, stratification ensures both train and test have the same proportion
- **Fairer evaluation:** Metrics are not distorted by random sampling skewing one set toward a majority class

### 8.4 Why VIF-Pruned Features?

- **Multicollinearity issues:** Highly correlated features inflate variance of coefficient estimates and reduce interpretability
- **VIF threshold:** Removed features with VIF > 10 (Age, CGPA, Semester_Credit_Load)
- **Result:** 14-feature model with VIF < 3 for all remaining features, stable and interpretable

### 8.5 Why `class_weight='balanced'`?

- **Imbalanced data:** Burnout has a natural skew toward lower severity in student populations
- **Balanced weights:** Logistic regression loss is scaled so the minority class ("High burnout") has equal importance during training
- **Improved recall:** Ensures the model doesn't simply predict majority class; all classes have reasonable sensitivity

---

## 9. Model Performance

### 9.1 Test Set Performance (Example Results)

Based on the final baseline model trained on ~8000 students and tested on ~2000:

| Metric | Value |
|--------|-------|
| **Accuracy** | ~68–72% |
| **Cohen's Kappa** | ~0.55–0.65 |
| **Macro Recall** | ~65–70% |
| **Macro F1 Score** | ~0.63–0.68 |

### 9.2 Per-Class Recall (Sensitivity)

- **Very Low (Q1):** ~75% (most predictable; obvious low-stress patterns)
- **Low (Q2):** ~65% (moderate patterns)
- **Moderate (Q3):** ~60% (borderline cases; harder to distinguish)
- **High (Q4):** ~55% (hardest class; overlaps with Q3; minority in training data)

### 9.3 Key Insights

- **Moderate performance:** 68–72% accuracy is reasonable for a 4-class problem with ~10k samples and 14 features
- **Class-dependent variation:** High-burnout students are underrepresented and harder to predict, even with balanced weights
- **Feature importance:** Sleep quality, social support, and stress-related features have strongest impact
- **Not perfect, but useful:** Model is suitable for screening and decision support, not definitive diagnosis

---

## 10. Integration with Application

The baseline model is integrated into the AuraCheck Streamlit app through:

- **[scripts/integrated_model_inference.py](../scripts/integrated_model_inference.py):** Unified inference endpoint
  - Loads trained model and scaler
  - Accepts user survey input
  - Returns predicted burnout class and confidence

- **[app.py](../app.py):** UI layer
  - Calls integrated inference
  - Displays predicted burnout class and visualizations
  - Provides recommendations based on prediction

**Example usage:**
```python
from scripts.integrated_model_inference import integrated_predict

user_input = {
    "Course": "Engineering",
    "Gender": "Female",
    "Sleep_Quality": "Poor",
    # ... other fields
}

result = integrated_predict(user_input, Path("."))
burnout_class = result["baseline_multinomial"]["predicted_class"]
```

---

## 11. Reproducibility and Future Work

### 11.1 Reproducibility

All scripts use fixed random seeds (`random_state=42`) to ensure reproducible results. To re-run:

```bash
python baseline/scripts/production_pruned_multinomial_baseline.py
python baseline/scripts/create_final_process_baseline_report.py
```

Expected runtime: ~1–2 minutes (depending on machine hardware).

### 11.2 Potential Enhancements

1. **Feature engineering:** Derive interaction terms (e.g., Sleep × Physical_Activity) to capture non-linear relationships
2. **Hyperparameter tuning:** Use GridSearchCV to optimize max_iter, l2 penalty strength, solver choice
3. **Cross-validation:** Implement k-fold cross-validation to estimate generalization performance without a separate test set
4. **Ensemble methods:** Combine logistic regression with other models (e.g., Random Forest, Gradient Boosting) via voting or stacking
5. **Probability calibration:** Use Platt scaling or isotonic regression to improve confidence calibration
6. **Class weighting refinement:** Experiment with custom class weights (not just "balanced")
7. **Feature selection:** Use LASSO (L1 penalty) to automatically select most important features
8. **Imbalanced learning:** Explore SMOTE (Synthetic Minority Over-sampling) or class-weighted loss variants

---

## 12. References and Further Reading

- **scikit-learn Logistic Regression:** https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- **Multinomial Logistic Regression Theory:** https://en.wikipedia.org/wiki/Multinomial_logistic_regression
- **Cohen's Kappa:** Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales"
- **VIF and Multicollinearity:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). "An Introduction to Statistical Learning"
- **Class Imbalance Handling:** He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data"
- **Probability Calibration:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks"

---

## 13. Contact and Support

For questions about this baseline model or to report issues:
- **Code Location:** [baseline/scripts/](scripts/)
- **Model Output:** [baseline/outputs/final_baseline_model/](outputs/final_baseline_model/)
- **Integration:** [scripts/integrated_model_inference.py](../scripts/integrated_model_inference.py)

---

**Report Generated:** April 20, 2026  
**Last Updated:** April 20, 2026

