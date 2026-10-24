"""
Customer Churn Prediction System - Phase 2: Model Training and Comparison
Script: train_models.py

Description:
Trains, compares, and evaluates three supervised classification models
(Logistic Regression, Random Forest, Neural Network/MLP) to predict customer churn.
Selects the best model based primarily on F1 Score, saves the model artifact,
and generates performance visualizations and metric summaries.
"""

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================================
# 1. DATA VALIDATION & LOADING
# =====================================================================

def validate_and_load_data(file_path: str = "data/customer_churn.csv") -> pd.DataFrame:
    """
    Validate dataset existence, schema completeness, and value domains.
    Exits with clear error message if any validation fails.
    """
    print("=" * 65)
    print("1. DATA LOADING & VALIDATION")
    print("=" * 65)

    # Check 1: File existence
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found at: {file_path}")
        print("Please ensure Phase 1 data generation has completed: `python generate_data.py`")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print(f"Loaded dataset successfully from '{file_path}'. Shape: {df.shape[0]} rows x {df.shape[1]} columns.")

    # Check 2: Required columns present
    required_features = [
        "tenure_months",
        "monthly_charge",
        "support_tickets",
        "contract_months",
        "autopay",
        "usage_hours"
    ]
    target_col = "churn"
    expected_cols = set(required_features + [target_col])
    missing_cols = expected_cols - set(df.columns)

    if missing_cols:
        print(f"[ERROR] Missing required columns in dataset: {missing_cols}")
        sys.exit(1)

    # Check 3: Churn target contains strictly binary values {0, 1}
    unique_churn = set(df[target_col].unique())
    if not unique_churn.issubset({0, 1}):
        print(f"[ERROR] Target column '{target_col}' must contain only 0 and 1. Found: {unique_churn}")
        sys.exit(1)

    print("[PASS] Dataset file exists, all required columns are present, and target contains only {0, 1}.")
    return df


# =====================================================================
# 2. TRAIN-TEST SPLITTING
# =====================================================================

def prepare_data_split(df: pd.DataFrame):
    """
    Extract features and target, and split into 80% train and 20% test
    stratified by churn to preserve class distribution.
    """
    print("\n" + "=" * 65)
    print("2. TRAIN-TEST SPLIT (80% Train, 20% Test, Stratified)")
    print("=" * 65)

    feature_cols = [
        "tenure_months",
        "monthly_charge",
        "support_tickets",
        "contract_months",
        "autopay",
        "usage_hours"
    ]
    target_col = "churn"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training set size: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test set size:     {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

    # Class distributions
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100

    print("\nClass Distribution in Training Set:")
    print(f"  Class 0 (Stayed):  {y_train.value_counts()[0]} ({train_dist[0]:.2f}%)")
    print(f"  Class 1 (Churned): {y_train.value_counts()[1]} ({train_dist[1]:.2f}%)")

    print("\nClass Distribution in Test Set:")
    print(f"  Class 0 (Stayed):  {y_test.value_counts()[0]} ({test_dist[0]:.2f}%)")
    print(f"  Class 1 (Churned): {y_test.value_counts()[1]} ({test_dist[1]:.2f}%)")

    return X_train, X_test, y_train, y_test


# =====================================================================
# 3. MODEL DEFINITION & TRAINING
# =====================================================================

def build_and_train_models(X_train, y_train, X_test, y_test):
    """
    Build, train, and evaluate the three required classification models.
    Captures any warnings encountered during training.
    """
    print("\n" + "=" * 65)
    print("3. MODEL TRAINING & EVALUATION")
    print("=" * 65)

    # Define model dictionary with pipelines where scaling is required
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "Neural Network (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                random_state=42
            ))
        ])
    }

    metrics_list = []
    trained_models = {}
    test_predictions = {}
    encountered_warnings = []

    for name, model in models.items():
        print(f"Training {name}...")

        # Capture warnings during fit
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)

            for w in captured_warnings:
                msg = f"{name}: {w.category.__name__} - {w.message}"
                encountered_warnings.append(msg)

        trained_models[name] = model

        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred

        # ROC AUC requires predicted probabilities for positive class (1)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = float("nan")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

    metrics_df = pd.DataFrame(metrics_list)
    return trained_models, metrics_df, test_predictions, encountered_warnings


# =====================================================================
# 4. MODEL SELECTION & RATIONALE
# =====================================================================

def select_best_model(metrics_df: pd.DataFrame):
    """
    Select the best performing model based primarily on F1 Score.
    """
    best_row = metrics_df.loc[metrics_df["F1 Score"].idxmax()]
    best_name = best_row["Model"]
    best_f1 = best_row["F1 Score"]
    return best_name, best_f1, best_row


# =====================================================================
# 5. VISUALIZATION & ARTIFACT GENERATION
# =====================================================================

def save_visualizations_and_artifacts(
    best_model_name: str,
    trained_models: dict,
    metrics_df: pd.DataFrame,
    test_predictions: dict,
    y_test: pd.Series,
    output_dir: str = "artifacts"
):
    """
    Save the winning model, comparison table, confusion matrix, and F1 comparison bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save winning model
    model_path = os.path.join(output_dir, "churn_model.joblib")
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, model_path)

    # 2. Save comparison metrics to CSV
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # 3. Create & save Confusion Matrix for the best model
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    best_pred = test_predictions[best_model_name]
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed (0)", "Churned (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(f"Confusion Matrix: Best Model ({best_model_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="medium")
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="medium")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)

    # 4. Create & save F1 Score comparison bar chart
    f1_chart_path = os.path.join(output_dir, "f1_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 4.8))

    models = metrics_df["Model"].tolist()
    f1_scores = metrics_df["F1 Score"].tolist()

    # Highlight winning model with a standout color
    colors = ["#10b981" if m == best_model_name else "#64748b" for m in models]
    bars = ax.bar(models, f1_scores, color=colors, width=0.45, edgecolor="#0f172a", linewidth=1.2)

    # Add score annotations above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylim(0, max(f1_scores) * 1.25)
    ax.set_ylabel("F1 Score (Harmonic Mean of Precision & Recall)", fontsize=10, fontweight="medium")
    ax.set_title("Model Comparison: Test Set F1 Score", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend / Annotation for winner
    plt.legend([bars[models.index(best_model_name)]], [f"Winner: {best_model_name}"], loc="upper right")
    plt.tight_layout()
    plt.savefig(f1_chart_path, dpi=300)
    plt.close(fig)

    return model_path, metrics_path, cm_path, f1_chart_path, cm


# =====================================================================
# 6. MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION SYSTEM - PHASE 2: MODEL TRAINING")
    print("=" * 65)

    # Step 1: Validate & Load
    df = validate_and_load_data("data/customer_churn.csv")

    # Step 2: Split
    X_train, X_test, y_train, y_test = prepare_data_split(df)

    # Step 3: Train & Evaluate
    trained_models, metrics_df, test_predictions, encountered_warnings = build_and_train_models(
        X_train, y_train, X_test, y_test
    )

    # Step 4: Display Comparison Table
    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE COMPARISON (TEST SET)")
    print("=" * 65)
    formatted_table = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        formatted_table[col] = formatted_table[col].map(lambda x: f"{x:.4f}")
    print(formatted_table.to_string(index=False))

    # Step 5: Select Best Model
    best_name, best_f1, best_row = select_best_model(metrics_df)
    print("\n" + "=" * 65)
    print("BEST MODEL SELECTION")
    print("=" * 65)
    print(f"Selected Best Model: {best_name} (F1 Score: {best_f1:.4f})")
    print("\nWhy F1 Score is the primary selection metric for Churn Classification:")
    print("  1. Class Imbalance: In churn prediction, customers who leave represent a minority")
    print("     class (~30% churn vs. ~70% stay). A naive baseline model predicting that every")
    print("     customer stays would achieve ~70% accuracy while failing to detect any churning customers.")
    print("  2. Balanced Trade-off: F1 Score is the harmonic mean of Precision and Recall.")
    print("     - Precision penalizes False Positives (wasting customer retention incentives on loyal users).")
    print("     - Recall penalizes False Negatives (missing at-risk customers who cancel without intervention).")
    print("  3. Reliability: Selecting the highest F1 score ensures the business catches at-risk accounts")
    print("     efficiently without flooding operations with false alarms.")

    # Step 6: Save Artifacts & Visualizations
    model_path, metrics_path, cm_path, f1_chart_path, cm = save_visualizations_and_artifacts(
        best_name, trained_models, metrics_df, test_predictions, y_test
    )

    # Step 7: Confusion Matrix Summary
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "=" * 65)
    print(f"CONFUSION MATRIX SUMMARY ({best_name})")
    print("=" * 65)
    print(f"  True Negatives  (TN - Correctly predicted stayed):  {tn}")
    print(f"  False Positives (FP - Incorrectly predicted churn): {fp}")
    print(f"  False Negatives (FN - Missed churners who left):    {fn}")
    print(f"  True Positives  (TP - Correctly caught churners):   {tp}")
    print(f"  Total Test Samples:                                 {tn + fp + fn + tp}")

    # Step 8: Warnings Encountered
    print("\n" + "=" * 65)
    print("WARNINGS ENCOUNTERED DURING TRAINING")
    print("=" * 65)
    if encountered_warnings:
        for w in encountered_warnings:
            print(f"  [WARNING] {w}")
    else:
        print("  None. All models trained without warnings.")

    # Step 9: Saved Artifacts Summary
    print("\n" + "=" * 65)
    print("SAVED ARTIFACTS")
    print("=" * 65)
    print(f"  1. Best Model File:       {model_path}")
    print(f"  2. Model Metrics CSV:     {metrics_path}")
    print(f"  3. Confusion Matrix Plot: {cm_path}")
    print(f"  4. F1 Comparison Plot:    {f1_chart_path}")
    print("=" * 65)
    print("Phase 2 Model Training & Comparison successfully completed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
