"""
Customer Churn Prediction System - Phase 1: Data Preparation
Script: generate_data.py

Description:
Generates a reproducible synthetic dataset with 1,500 customer records
and saves it to data/customer_churn.csv. Features are designed to simulate
realistic relationships with customer churn:
- High support tickets & high monthly charges increase churn risk.
- Longer tenure, longer contracts, automatic payments, and higher usage reduce churn risk.
"""

import os
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer records with realistic churn behaviors.

    Parameters:
        n_samples (int): Number of customer records to generate (default: 1500).
        random_seed (int): Seed for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Generated dataset containing customer features and churn label.
    """
    # Set random seed for full reproducibility
    np.random.seed(random_seed)

    # 1. Feature: tenure_months (integer 1 to 72 months)
    tenure_months = np.random.randint(1, 73, size=n_samples)

    # 2. Feature: monthly_charge (continuous $20.00 to $150.00, rounded to 2 decimals)
    monthly_charge = np.round(np.random.uniform(20.0, 150.0, size=n_samples), 2)

    # 3. Feature: support_tickets (integer count, realistic Poisson distribution clipped to [0, 10])
    support_tickets = np.clip(np.random.poisson(lam=2.0, size=n_samples), 0, 10)

    # 4. Feature: contract_months (categorical duration: 1, 12, or 24 months)
    # Month-to-month contracts are most common, followed by 1-year and 2-year plans
    contract_months = np.random.choice([1, 12, 24], size=n_samples, p=[0.50, 0.30, 0.20])

    # 5. Feature: autopay (binary: 0 = manual payment, 1 = automated payment)
    autopay = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])

    # 6. Feature: usage_hours (continuous 5.0 to 100.0 hours/month, rounded to 1 decimal)
    usage_hours = np.round(np.random.uniform(5.0, 100.0, size=n_samples), 1)

    # --- Realistic Churn Relationship Calculation ---
    # Normalize features to [0, 1] range to apply balanced logistic weights
    tenure_norm = (tenure_months - 1.0) / 71.0
    monthly_charge_norm = (monthly_charge - 20.0) / 130.0
    support_tickets_norm = support_tickets / 10.0
    contract_norm = (contract_months - 1.0) / 23.0
    autopay_norm = autopay.astype(float)
    usage_norm = (usage_hours - 5.0) / 95.0

    # Calculate log-odds (logit) of churn:
    # (+) features that increase churn: support_tickets, monthly_charge
    # (-) features that reduce churn: tenure_months, contract_months, autopay, usage_hours
    logit = (
        0.2
        + 2.8 * support_tickets_norm
        + 1.8 * monthly_charge_norm
        - 2.2 * tenure_norm
        - 2.0 * contract_norm
        - 0.9 * autopay_norm
        - 1.2 * usage_norm
        + np.random.normal(0, 0.4, size=n_samples)  # realistic random noise
    )

    # Convert log-odds into probabilities using standard sigmoid function
    churn_probability = 1.0 / (1.0 + np.exp(-logit))

    # Sample binary outcome: 1 = customer churns, 0 = customer stays
    churn = (np.random.uniform(0.0, 1.0, size=n_samples) < churn_probability).astype(int)

    # Assemble into a clean pandas DataFrame
    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "support_tickets": support_tickets,
        "contract_months": contract_months,
        "autopay": autopay,
        "usage_hours": usage_hours,
        "churn": churn
    })

    return df


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION SYSTEM - DATA GENERATION")
    print("=" * 60)

    # Target directory and file path
    output_dir = "data"
    output_file = os.path.join(output_dir, "customer_churn.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating synthetic dataset (1,500 records, seed=42)...")
    df = generate_churn_dataset(n_samples=1500, random_seed=42)

    # Save to CSV without index
    df.to_csv(output_file, index=False)
    print(f"Dataset successfully created and saved to: {output_file}")
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
