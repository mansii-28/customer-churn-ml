"""
Customer Churn Prediction System - Phase 3: Inference & Edge-Case Testing
Script: test_inference.py

Description:
Validates model inference using saved artifacts/churn_model.joblib.
Runs realistic test scenarios (low vs high risk) and boundary edge cases
(min, max, mid values), verifying output schemas, probability constraints,
and risk categorization.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np


EXPECTED_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "contract_months",
    "autopay",
    "usage_hours"
]


def get_risk_category(probability: float) -> str:
    """
    Categorize churn risk based on estimated probability:
    - LOW RISK:    probability < 0.35
    - MEDIUM RISK: 0.35 <= probability < 0.65
    - HIGH RISK:   probability >= 0.65
    """
    if probability < 0.35:
        return "LOW RISK"
    elif probability < 0.65:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def load_model(model_path: str = "artifacts/churn_model.joblib"):
    """Load model artifact and verify file presence."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Please run `python train_models.py` before running inference tests.")
        sys.exit(1)
    return joblib.load(model_path)


def run_prediction(model, input_dict: dict) -> tuple:
    """
    Run prediction on a dictionary of features and validate pipeline execution.
    Returns (prediction, probability, risk_category, input_df).
    """
    input_df = pd.DataFrame([input_dict])

    # Validate feature presence and exact ordering
    assert list(input_df.columns) == EXPECTED_FEATURES, (
        f"Feature columns or order mismatch. Expected {EXPECTED_FEATURES}, got {list(input_df.columns)}"
    )

    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)

    # Validate shapes and probability boundaries
    assert pred.shape == (1,), f"Expected prediction shape (1,), got {pred.shape}"
    assert proba.shape == (1, 2), f"Expected proba shape (1, 2), got {proba.shape}"
    churn_prob = float(proba[0, 1])
    assert 0.0 <= churn_prob <= 1.0, f"Probability {churn_prob} outside [0.0, 1.0]"

    risk_cat = get_risk_category(churn_prob)
    return int(pred[0]), churn_prob, risk_cat, input_df


def test_core_scenarios(model):
    """
    Test Scenario A (lower churn profile) and Scenario B (higher churn profile).
    Verifies that Scenario B has higher estimated churn risk than Scenario A.
    """
    print("=" * 65)
    print("1. CORE BUSINESS SCENARIO TESTS")
    print("=" * 65)

    scenario_a = {
        "tenure_months": 60,
        "monthly_charge": 45.0,
        "support_tickets": 0,
        "contract_months": 24,
        "autopay": 1,
        "usage_hours": 80.0
    }

    scenario_b = {
        "tenure_months": 3,
        "monthly_charge": 140.0,
        "support_tickets": 7,
        "contract_months": 1,
        "autopay": 0,
        "usage_hours": 10.0
    }

    print("Running Scenario A (Expected Low Risk Customer)...")
    pred_a, proba_a, risk_a, _ = run_prediction(model, scenario_a)
    print(f"  Inputs:      {scenario_a}")
    print(f"  Prediction:  {pred_a} ({'Churn' if pred_a == 1 else 'Stay'})")
    print(f"  Probability: {proba_a:.4f} ({proba_a * 100:.2f}%)")
    print(f"  Risk Level:  {risk_a}")

    print("\nRunning Scenario B (Expected High Risk Customer)...")
    pred_b, proba_b, risk_b, _ = run_prediction(model, scenario_b)
    print(f"  Inputs:      {scenario_b}")
    print(f"  Prediction:  {pred_b} ({'Churn' if pred_b == 1 else 'Stay'})")
    print(f"  Probability: {proba_b:.4f} ({proba_b * 100:.2f}%)")
    print(f"  Risk Level:  {risk_b}")

    # Relative risk assertion
    assert proba_b > proba_a, (
        f"Assertion Failed: Expected Scenario B churn probability ({proba_b:.4f}) "
        f"to exceed Scenario A ({proba_a:.4f})"
    )
    print("\n[PASS] Relative Risk Check: High-risk scenario probability > Low-risk scenario probability.")

    return (pred_a, proba_a, risk_a), (pred_b, proba_b, risk_b)


def test_edge_cases(model):
    """
    Test model inference across extreme and middle-range boundary values:
    - Minimum allowed values
    - Maximum allowed values
    - Typical middle-range values
    """
    print("\n" + "=" * 65)
    print("2. EDGE-CASE & BOUNDARY TESTS")
    print("=" * 65)

    edge_cases = {
        "Minimum Allowed Values": {
            "tenure_months": 1,
            "monthly_charge": 20.0,
            "support_tickets": 0,
            "contract_months": 1,
            "autopay": 0,
            "usage_hours": 5.0
        },
        "Maximum Allowed Values": {
            "tenure_months": 72,
            "monthly_charge": 150.0,
            "support_tickets": 10,
            "contract_months": 24,
            "autopay": 1,
            "usage_hours": 100.0
        },
        "Typical Middle-Range Customer": {
            "tenure_months": 36,
            "monthly_charge": 85.0,
            "support_tickets": 2,
            "contract_months": 12,
            "autopay": 1,
            "usage_hours": 50.0
        }
    }

    for case_name, inputs in edge_cases.items():
        pred, proba, risk_cat, _ = run_prediction(model, inputs)
        print(f"Test '{case_name}':")
        print(f"  Inputs:      {inputs}")
        print(f"  Prediction:  {pred} ({'Churn' if pred == 1 else 'Stay'})")
        print(f"  Probability: {proba:.4f} ({proba * 100:.2f}%)")
        print(f"  Risk Level:  {risk_cat}")
        print(f"  Status:      [PASS]\n")


def main():
    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION SYSTEM - INFERENCE TESTING")
    print("=" * 65)

    model = load_model("artifacts/churn_model.joblib")
    test_core_scenarios(model)
    test_edge_cases(model)

    print("=" * 65)
    print("ALL INFERENCE & EDGE-CASE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
