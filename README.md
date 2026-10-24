# Customer Churn Prediction System

## Project Overview
The **Customer Churn Prediction System** is a portfolio-ready, end-to-end machine learning project designed to estimate the probability that subscription customers will cancel their service (churn). By detecting at-risk accounts in advance, customer success and marketing teams can deliver timely retention outreach, resolve underlying pain points, and protect recurring subscription revenue.

---

## Problem Type
**Supervised Binary Classification**:
- **Supervised**: Trained using labeled historical customer account records with known outcomes.
- **Binary Outcome**:
  - `0`: Customer stays with the service (Retained / Negative Class)
  - `1`: Customer cancels the service (Churned / Positive Class)

---

## Dataset
- **Cohort**: 1,500 synthetic customer records generated reproducibly (`random_seed=42`).
- **File Location**: `data/customer_churn.csv`
- **Class Balance**: 1,052 Retained (70.13%) vs. 448 Churned (29.87%).
- **Completeness**: 0 missing values, 0 duplicate records, 100% verified schema.

---

## Features

| Feature Name | Type | Value Range | Business Meaning & Churn Impact |
| :--- | :--- | :--- | :--- |
| `tenure_months` | Integer | `1` to `72` | Number of continuous subscription months. Longer tenure reflects loyalty and decreases churn risk. |
| `monthly_charge` | Float | `$20.00` to `$150.00` | Recurring monthly subscription fee in USD. Higher charges increase customer cost sensitivity and churn risk. |
| `support_tickets` | Integer | `0` to `10` | Total support inquiries submitted. Higher tickets indicate unresolved friction, increasing churn risk. |
| `contract_months` | Integer | `1`, `12`, or `24` | Subscription contract commitment duration. Longer contracts significantly reduce churn risk. |
| `autopay` | Integer | `0` or `1` | Automatic billing enabled (`1` = Yes, `0` = No). Autopay removes payment friction and lowers churn risk. |
| `usage_hours` | Float | `5.0` to `100.0` | Monthly platform active usage in hours. Higher usage indicates product engagement and reduces churn risk. |

---

## Phase Breakdown

### Phase 1 - Data Preparation and EDA
- **Synthetic Generation (`generate_data.py`)**: Produced 1,500 records using domain-informed logistic dynamics with realistic noise.
- **Exploratory Data Analysis (`explore_data.py`)**: Analyzed distributions, computed descriptive statistics, and validated expected feature correlations against `churn`.
- **Automated Validation**: Built an 8-point data integrity test suite verifying completeness, target labels, and numeric ranges.

### Phase 2 - Model Training and Evaluation
- **Stratified Split**: Partitioned data into 80% training (1,200 records) and 20% test (300 records) holding class proportions constant.
- **Algorithms Trained**: Logistic Regression (linear baseline), Random Forest (tree ensemble), and Multi-Layer Perceptron Neural Network.
- **Feature Scaling**: Implemented `StandardScaler` inside scikit-learn `Pipeline` objects for Logistic Regression and Neural Networks to ensure balanced gradient updates without test leakage.
- **Model Selection**: Selected **Random Forest** based primarily on **F1 Score** to balance precision and recall on imbalanced target classes.
- **Artifact Export**: Serialized the winning model to `artifacts/churn_model.joblib` and exported comparison charts.

### Phase 3 - Prediction App and Deployment
- **Interactive UI (`app.py`)**: Built a modern Streamlit web application providing sliders and inputs for all customer features.
- **Inference Pipeline**: Encapsulates model loading with error handling, schema order verification, and probability estimation.
- **Business Risk Tiering**: Converts raw probabilities into actionable tiers (Low, Medium, High Risk) accompanied by non-deterministic interpretation.
- **Explainability**: Integrated global Random Forest feature importances (`feature_importance.py`) into the application.
- **Quality Assurance (`test_inference.py`)**: Tested standard business scenarios, relative risk consistency, and boundary edge cases.

---

## Models Compared

| Model Architecture | Description | Pipeline Preprocessing |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear probabilistic classifier with L2 regularization (`max_iter=1000`) | `StandardScaler()` + `LogisticRegression()` |
| **Random Forest** *(Winner)* | Bagging ensemble of 200 decision trees (`n_estimators=200`) | None (Trees are scale-invariant) |
| **Neural Network (MLP)** | Multi-Layer Perceptron architecture with hidden layers `(32, 16)` (`max_iter=1000`) | `StandardScaler()` + `MLPClassifier()` |

---

## Evaluation Metrics

Models were benchmarked on the identical held-out 20% test set (300 samples: 210 Stayed, 90 Churned):

| Evaluation Metric | Business Purpose |
| :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions across all classes. |
| **Precision** | Accuracy of positive churn predictions ($\frac{TP}{TP+FP}$). Controls wasteful retention spend on happy customers. |
| **Recall** | Coverage of true churners detected ($\frac{TP}{TP+FN}$). Controls revenue loss from uncontacted churners. |
| **F1 Score** | Harmonic mean balancing Precision and Recall. Primary selection metric due to class imbalance. |
| **ROC AUC** | Model discriminative capability across all decision threshold values. |

### Test Set Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** *(Selected)* | **74.67%** | **60.94%** | **43.33%** | **0.5065** | **0.7141** |
| **Logistic Regression** | 74.33% | 61.40% | 38.89% | 0.4762 | 0.7517 |
| **Neural Network (MLP)** | 65.33% | 41.46% | 37.78% | 0.3953 | 0.6255 |

---

## Best Model: Random Forest Classifier

- **Primary Criterion**: Achieved the highest **F1 Score (0.5065)** and highest test accuracy (74.67%).
- **Confusion Matrix Summary** (300 Test Samples):
  - **True Negatives (TN)**: `185` (Correctly predicted customer stays)
  - **False Positives (FP)**: `25` (False alarm: loyal customer flagged as churner)
  - **False Negatives (FN)**: `51` (Missed churners who canceled)
  - **True Positives (TP)**: `39` (Successfully intercepted churners)

### Global Feature Importance Ranking

| Rank | Feature | Description | Importance Share |
| :---: | :--- | :--- | :---: |
| **1** | `monthly_charge` | Monthly Billing Amount ($) | **27.72%** |
| **2** | `usage_hours` | Monthly Platform Usage (Hours) | **24.98%** |
| **3** | `tenure_months` | Customer Lifetime Tenure (Months) | **24.47%** |
| **4** | `support_tickets` | Customer Support Inquiries | **10.33%** |
| **5** | `contract_months` | Contract Duration (Months) | **8.55%** |
| **6** | `autopay` | Automatic Billing Status | **3.95%** |

---

## Project Structure

```text
customer-churn-ml/
├── data/
│   └── customer_churn.csv         # Synthetic customer dataset (1,500 rows x 7 cols)
├── artifacts/
│   ├── .gitkeep                    # Directory tracking
│   ├── churn_model.joblib          # Trained Random Forest model artifact
│   ├── model_metrics.csv           # Model benchmark metrics table
│   ├── confusion_matrix.png        # Confusion matrix visualization
│   ├── f1_comparison.png           # F1 score comparison bar chart
│   ├── feature_importance.csv      # Ranked feature importances data
│   └── feature_importance.png      # Feature importance bar chart
├── generate_data.py                # Phase 1: Synthetic data generator
├── explore_data.py                 # Phase 1: EDA & automated data validation
├── train_models.py                 # Phase 2: Model training, evaluation & export
├── test_inference.py               # Phase 3: Inference & edge-case test suite
├── feature_importance.py           # Phase 3: Feature importance calculation & plot
├── app.py                          # Phase 3: Interactive Streamlit web application
├── requirements.txt                # Project Python dependencies
├── .gitignore                      # Git ignore file for environment & cache
└── README.md                       # Complete project documentation
```

---

## Installation

```bash
# 1. Clone or navigate to the repository
cd customer-churn-ml

# 2. Create and activate a Python 3.9+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## How to Run the Project

### Step 1: Data Preparation & Validation
```bash
# Generate the reproducible dataset
python generate_data.py

# Run exploratory data analysis and validation checks
python explore_data.py
```

### Step 2: Model Training & Evaluation
```bash
# Train all models, evaluate metrics, and save the winning artifact
python train_models.py
```

### Step 3: Feature Importance Analysis
```bash
# Extract and plot Random Forest feature importances
python feature_importance.py
```

### Step 4: Run Inference & Edge-Case Tests
```bash
# Execute unit validation and scenario tests
python test_inference.py
```

### Step 5: Launch the Streamlit Prediction Web App
```bash
# Start the local Streamlit application server
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Example Prediction Workflow

1. **User enters customer profile** in the Streamlit UI:
   - Tenure: `3 months`
   - Monthly Charge: `$140.00`
   - Support Tickets: `7`
   - Contract: `1 Month (Month-to-month)`
   - Autopay: `No`
   - Monthly Usage: `10.0 hours`
2. **User clicks "Predict Churn Risk"**.
3. **Model evaluates inputs**:
   - Estimated Churn Probability: `89.0%`
   - Risk Category: **HIGH RISK (🔴)**
   - Recommendation: *"This customer appears at higher risk of churn and may be a candidate for retention outreach."*
4. **Business takes proactive action**:
   - Customer success initiates priority outreach to resolve the 7 support issues and suggests switching to a discounted annual contract with autopay.

---

## Limitations
- **Synthetic Data**: The data was generated through statistical simulation and does not contain the full complexity or subtle anomalies of real customer telemetry.
- **Educational Scope**: Designed as an educational demonstration of clean ML engineering patterns.
- **Moderate Performance**: The winning model achieves a 0.5065 F1 score with 43.33% recall on default 0.50 probability thresholds.
- **No Real-World Validation**: Not validated on live customer portfolios or external business contexts.
- **No Fairness / Bias Auditing**: No demographic or subgroup fairness audits have been conducted.
- **No Production Monitoring**: Does not include data drift monitoring, automated retraining triggers, or telemetry logging.

---

## Future Improvements
- **Hyperparameter Optimization**: Conduct Bayesian optimization (e.g., Optuna) or extensive grid search over tree depth, split criteria, and minimum leaf samples.
- **Threshold Optimization**: Tune the classification decision threshold to optimize for business cost matrices (e.g., weighing cost of an intervention against customer lifetime value).
- **Cross-Validation**: Implement k-fold cross-validation (e.g., 5-fold Stratified K-Fold) for more robust variance estimation.
- **Class Imbalance Mitigation**: Test SMOTE (Synthetic Minority Over-sampling Technique) or cost-sensitive class weights (`class_weight='balanced'`).
- **SHAP Explainability**: Incorporate TreeSHAP to provide localized feature attribution waterfall plots for each individual prediction.
- **Real-World Data**: Ingest real industry datasets (such as Telco Churn benchmarks) for production transferability.
- **Cloud Deployment**: Containerize with Docker and deploy to cloud platforms (e.g., Streamlit Community Cloud, AWS App Runner, GCP Cloud Run).
