"""
Customer Churn Prediction System - Phase 3: Streamlit Application
Script: app.py

Description:
Interactive web application for real-time customer churn risk assessment.
Loads the trained Random Forest model artifact, takes customer features,
estimates churn probability, maps to business risk tiers, and displays
model evaluation metrics and feature importances.
"""

import os
import joblib
import pandas as pd
import streamlit as st


# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

EXPECTED_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "contract_months",
    "autopay",
    "usage_hours"
]

MODEL_PATH = "artifacts/churn_model.joblib"
METRICS_PATH = "artifacts/model_metrics.csv"
CONFUSION_MATRIX_PATH = "artifacts/confusion_matrix.png"
F1_COMPARISON_PATH = "artifacts/f1_comparison.png"
IMPORTANCE_CSV_PATH = "artifacts/feature_importance.csv"
IMPORTANCE_PNG_PATH = "artifacts/feature_importance.png"


# =====================================================================
# CACHED MODEL LOADING & VALIDATION
# =====================================================================

@st.cache_resource
def load_churn_model(path: str = MODEL_PATH):
    """Load model artifact with validation and caching."""
    if not os.path.exists(path):
        return None, f"Model file not found at '{path}'. Please run `python train_models.py` first."
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model artifact: {str(e)}"


# =====================================================================
# MAIN APPLICATION INTERFACE
# =====================================================================

def main():
    # Header & Introduction
    st.title("📊 Customer Churn Prediction System")
    st.markdown(
        """
        Welcome to the **Customer Churn Prediction System**. This tool estimates the **probability 
        that an active subscription customer may cancel their service (churn)**.

        Enter a customer's account characteristics in the input form below and click **Predict Churn Risk** 
        to evaluate their risk level and view recommended retention guidance.
        """
    )
    st.divider()

    # Load Model
    model, load_error = load_churn_model()
    if load_error:
        st.error(f"⚠️ **Model Loading Error**: {load_error}")
        st.info("Ensure that Phase 2 model training has completed by executing: `python train_models.py`")
        return

    # Two-column layout: Inputs on left/main, Results on right/main
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📋 Customer Account Profile")
        st.caption("Adjust the sliders and dropdowns below to reflect the customer's attributes.")

        with st.form("churn_prediction_form"):
            # Feature 1: tenure_months (1 to 72)
            tenure_months = st.slider(
                "Customer Tenure (Months)",
                min_value=1,
                max_value=72,
                value=24,
                step=1,
                help="Total continuous months the customer has been subscribed (1 to 72 months)."
            )

            # Feature 2: monthly_charge ($20 to $150)
            monthly_charge = st.slider(
                "Monthly Charge ($)",
                min_value=20.0,
                max_value=150.0,
                value=70.0,
                step=1.0,
                format="$%.2f",
                help="Recurring monthly billing amount in USD ($20 to $150)."
            )

            # Feature 3: support_tickets (0 to 10)
            support_tickets = st.number_input(
                "Support Tickets Submitted",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                help="Total customer service inquiries or support tickets submitted (0 to 10)."
            )

            # Feature 4: contract_months (1, 12, or 24)
            contract_options = {
                "Month-to-Month (1 Month)": 1,
                "One Year (12 Months)": 12,
                "Two Years (24 Months)": 24
            }
            contract_selection = st.selectbox(
                "Contract Duration",
                options=list(contract_options.keys()),
                index=1,
                help="Length of subscription agreement: 1, 12, or 24 months."
            )
            contract_months = contract_options[contract_selection]

            # Feature 5: autopay (Yes / No -> 1 / 0)
            autopay_selection = st.radio(
                "Automatic Payment (Autopay) Enabled?",
                options=["Yes", "No"],
                index=0,
                horizontal=True,
                help="Whether automated billing / payment is activated."
            )
            autopay = 1 if autopay_selection == "Yes" else 0

            # Feature 6: usage_hours (5 to 100)
            usage_hours = st.slider(
                "Average Monthly Usage (Hours)",
                min_value=5.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help="Estimated platform or service usage time per month (5 to 100 hours)."
            )

            submit_btn = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

    with col_results:
        st.subheader("🎯 Risk Assessment & Recommendations")

        if submit_btn:
            # Build 1-row DataFrame with strict column order
            input_dict = {
                "tenure_months": tenure_months,
                "monthly_charge": float(monthly_charge),
                "support_tickets": int(support_tickets),
                "contract_months": int(contract_months),
                "autopay": int(autopay),
                "usage_hours": float(usage_hours)
            }
            input_df = pd.DataFrame([input_dict])[EXPECTED_FEATURES]

            # Validation check
            if list(input_df.columns) != EXPECTED_FEATURES:
                st.error("Validation Error: Input features do not match expected model schema.")
                return

            try:
                # Predict
                prediction = int(model.predict(input_df)[0])
                churn_prob = float(model.predict_proba(input_df)[0, 1])
                churn_pct = churn_prob * 100.0

                # Display Metric Callout
                st.metric(
                    label="Estimated Churn Probability",
                    value=f"{churn_pct:.1f}%",
                    help="Calculated by the ensemble Random Forest decision trees."
                )

                # Risk Categorization & Business Guidance
                if churn_prob < 0.35:
                    st.success("🟢 **LOW RISK**")
                    st.write(
                        "**Business Interpretation**: This customer appears relatively stable based on "
                        "the model inputs. Continue standard relationship maintenance without immediate intervention."
                    )
                elif churn_prob < 0.65:
                    st.warning("🟡 **MEDIUM RISK**")
                    st.write(
                        "**Business Interpretation**: This customer shows some churn risk and may benefit "
                        "from proactive engagement. Consider scheduling a check-in or offering tailored support."
                    )
                else:
                    st.error("🔴 **HIGH RISK**")
                    st.write(
                        "**Business Interpretation**: This customer appears at higher risk of churn and "
                        "may be a candidate for retention outreach. Prioritize account outreach, address pending "
                        "support issues, or explore loyalty incentives."
                    )

                # Responsible ML Disclaimer
                st.caption(
                    "⚠️ *Note: This prediction is a model-estimated statistical probability, not a deterministic certainty. "
                    "Predictions should inform, but not replace, human business judgment.*"
                )

                # Input Summary Table
                st.markdown("#### 📋 Customer Inputs Summary")
                summary_data = [
                    {"Customer Attribute": "Tenure", "Entered Value": f"{tenure_months} months"},
                    {"Customer Attribute": "Monthly Charge", "Entered Value": f"${monthly_charge:.2f}"},
                    {"Customer Attribute": "Support Tickets", "Entered Value": f"{support_tickets} ticket(s)"},
                    {"Customer Attribute": "Contract Duration", "Entered Value": f"{contract_months} months"},
                    {"Customer Attribute": "Automatic Billing (Autopay)", "Entered Value": autopay_selection},
                    {"Customer Attribute": "Monthly Usage", "Entered Value": f"{usage_hours:.1f} hours"}
                ]
                st.table(pd.DataFrame(summary_data))

            except Exception as e:
                st.error(f"Inference Error: An unexpected error occurred during prediction: {e}")

        else:
            st.info("👈 Configure the customer profile on the left and click **Predict Churn Risk** to generate an assessment.")

    st.divider()

    # =====================================================================
    # EXPANDABLE SECTIONS: MODEL INFO, EVALUATION, FEATURE IMPORTANCE
    # =====================================================================

    # Section 1: About the Model
    with st.expander("ℹ️ About the Model"):
        st.markdown(
            """
            ### Production Model Summary
            - **Selected Algorithm**: **Random Forest Classifier** (`n_estimators=200`, `random_state=42`)
            - **Training Cohort**: Synthetic dataset comprising **1,500 customer records**
            - **Train/Test Split**: **80/20 Stratified Split** (1,200 training samples / 300 held-out test samples)
            - **Model Selection Criterion**: Selected based on **F1 Score** to properly address class imbalance (~70% retained, ~30% churned)

            ### Held-Out Test Set Performance
            - **F1 Score**: `0.5065`
            - **ROC AUC**: `0.7141`
            - **Accuracy**: `74.67%`
            - **Precision**: `60.94%`
            - **Recall**: `43.33%`

            > *Disclaimer: This project is an educational machine learning demonstration using synthetic data and should not be used for real business decisions without validation on real-world data.*
            """
        )

    # Section 2: Model Evaluation
    with st.expander("📈 Model Evaluation & Comparison"):
        st.markdown("### Benchmark Against Alternative Supervised Models")
        st.write(
            "Three models were trained and benchmarked on the identical stratified test partition. "
            "Random Forest achieved the best balance of Precision and Recall as captured by F1 Score."
        )

        # 1. Metrics Table
        if os.path.exists(METRICS_PATH):
            try:
                metrics_df = pd.read_csv(METRICS_PATH)
                st.dataframe(metrics_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to read metrics table: {e}")
        else:
            st.info("Model metrics table not found at `artifacts/model_metrics.csv`.")

        # 2. Charts side-by-side
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if os.path.exists(F1_COMPARISON_PATH):
                st.image(F1_COMPARISON_PATH, caption="F1 Score Comparison Across Models", use_container_width=True)
            else:
                st.info("F1 comparison chart not found at `artifacts/f1_comparison.png`.")

        with chart_col2:
            if os.path.exists(CONFUSION_MATRIX_PATH):
                st.image(CONFUSION_MATRIX_PATH, caption="Confusion Matrix: Winning Random Forest Model", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found at `artifacts/confusion_matrix.png`.")

    # Section 3: Feature Importance
    with st.expander("🔍 Feature Importance"):
        st.markdown("### Random Forest Feature Importance Rankings")
        st.write(
            "Feature importance shows how much each input contributed to the Random Forest's "
            "decisions overall across all 200 bagging trees. It does not explain an individual prediction."
        )

        imp_col1, imp_col2 = st.columns([1, 1])

        with imp_col1:
            if os.path.exists(IMPORTANCE_CSV_PATH):
                try:
                    imp_df = pd.read_csv(IMPORTANCE_CSV_PATH)
                    display_cols = [c for c in ["Rank", "Friendly_Name", "Importance_Percent"] if c in imp_df.columns]
                    st.dataframe(imp_df[display_cols], use_container_width=True)
                except Exception as e:
                    st.warning(f"Unable to read feature importance table: {e}")
            else:
                st.info("Feature importance data not found at `artifacts/feature_importance.csv`.")

        with imp_col2:
            if os.path.exists(IMPORTANCE_PNG_PATH):
                st.image(IMPORTANCE_PNG_PATH, caption="Relative Feature Importance (%)", use_container_width=True)
            else:
                st.info("Feature importance chart not found at `artifacts/feature_importance.png`.")


if __name__ == "__main__":
    main()
