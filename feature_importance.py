"""
Customer Churn Prediction System - Phase 3: Feature Importance Analysis
Script: feature_importance.py

Description:
Loads the trained model (artifacts/churn_model.joblib), extracts global
feature importances from the Random Forest, ranks features from highest
to lowest impact, exports metrics to CSV, and generates an illustrative bar chart.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "contract_months",
    "autopay",
    "usage_hours"
]

FEATURE_LABELS = {
    "tenure_months": "Tenure (Months)",
    "monthly_charge": "Monthly Charge ($)",
    "support_tickets": "Support Tickets",
    "contract_months": "Contract Duration (Months)",
    "autopay": "Autopay (Automatic Billing)",
    "usage_hours": "Monthly Usage (Hours)"
}


def load_model(model_path: str = "artifacts/churn_model.joblib"):
    """Load model artifact with error validation."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model artifact not found at: {model_path}")
        print("Please run `python train_models.py` first to train and save the model.")
        sys.exit(1)

    model = joblib.load(model_path)
    return model


def extract_feature_importances(model, feature_names=FEATURE_NAMES) -> pd.DataFrame:
    """
    Extract and rank feature importances from the saved model.
    Handles raw tree estimators as well as scikit-learn Pipeline objects.
    """
    # If the model is wrapped inside a Pipeline, extract the final classifier
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("classifier", list(model.named_steps.values())[-1])

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(f"Estimator of type {type(estimator)} does not expose 'feature_importances_'.")

    importances = estimator.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: model has {len(importances)} features, expected {len(feature_names)}."
        )

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Friendly_Name": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "Importance": importances,
        "Importance_Percent": importances * 100.0
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    df_importance["Rank"] = range(1, len(df_importance) + 1)
    return df_importance


def generate_importance_plot(df_importance: pd.DataFrame, output_path: str = "artifacts/feature_importance.png"):
    """Generate and save a styled horizontal bar chart of feature importances."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Invert order for horizontal bar chart (top ranked at top)
    plot_df = df_importance.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        plot_df["Friendly_Name"],
        plot_df["Importance_Percent"],
        color="#2563eb",
        edgecolor="#1e3a8a",
        height=0.6,
        alpha=0.9
    )

    # Annotate percentage on each bar
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f" {width:.2f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#0f172a"
        )

    ax.set_xlim(0, max(plot_df["Importance_Percent"]) * 1.25)
    ax.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="medium")
    ax.set_title("Random Forest Global Feature Importance", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION SYSTEM - FEATURE IMPORTANCE ANALYSIS")
    print("=" * 65)

    model_path = "artifacts/churn_model.joblib"
    csv_path = "artifacts/feature_importance.csv"
    plot_path = "artifacts/feature_importance.png"

    model = load_model(model_path)
    df_importance = extract_feature_importances(model, FEATURE_NAMES)

    # Save to CSV
    os.makedirs("artifacts", exist_ok=True)
    df_importance.to_csv(csv_path, index=False)
    print(f"Saved feature importance table to: {csv_path}")

    # Generate plot
    generate_importance_plot(df_importance, plot_path)
    print(f"Saved feature importance plot to:  {plot_path}")

    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE RANKING (Random Forest)")
    print("=" * 65)
    display_df = df_importance[["Rank", "Feature", "Friendly_Name", "Importance_Percent"]].copy()
    display_df["Importance_Percent"] = display_df["Importance_Percent"].map(lambda x: f"{x:.2f}%")
    print(display_df.to_string(index=False))
    print("=" * 65)
    print("Feature importance analysis completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
