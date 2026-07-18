"""
feature_engineering.py
=======================
Must exactly mirror the transformations applied during training
(model_building notebook / chat pipeline). If this drifts from
training-time logic, predictions will be silently wrong.
"""

import numpy as np
import pandas as pd


NIGHT_HOURS = [19, 21, 22, 2, 4, 5]
HIGH_VALUE_THRESHOLD = 200_000


def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Takes a single raw transaction (as a dict) and returns a
    single-row DataFrame with all engineered features, matching
    training-time preprocessing exactly.

    Expected raw fields:
        kyc_verified: "Yes" | "No"
        account_age_days: int
        transaction_amount: float
        channel: "ATM" | "Mobile" | "POS" | "Web"
        timestamp: ISO datetime string
    """
    df = pd.DataFrame([raw])

    # --- timestamp -> hour_of_day, day_of_week ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # --- kyc_verified: Yes/No -> 1/0 ---
    df["kyc_verified"] = df["kyc_verified"].map({"Yes": 1, "No": 0})

    # --- channel: one-hot, drop_first=True (ATM was the dropped baseline in training) ---
    for ch in ["Mobile", "POS", "Web"]:
        df[f"channel_{ch}"] = (df["channel"] == ch).astype(int)

    # --- transaction_amount_log ---
    df["transaction_amount_log"] = np.log1p(df["transaction_amount"])

    # --- is_high_value ---
    df["is_high_value"] = (df["transaction_amount"] > HIGH_VALUE_THRESHOLD).astype(int)

    # --- is_night_txn ---
    df["is_night_txn"] = df["hour_of_day"].isin(NIGHT_HOURS).astype(int)

    # --- no_kyc_digital: unverified KYC + (Mobile or Web) ---
    df["no_kyc_digital"] = (
        (df["kyc_verified"] == 0)
        & ((df["channel_Mobile"] == 1) | (df["channel_Web"] == 1))
    ).astype(int)

    # drop raw columns not used directly as model input
    df = df.drop(columns=["channel", "timestamp"])

    return df


def align_to_training_columns(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Ensures the engineered dataframe has exactly the columns the model
    was trained on, in the exact same order. Missing columns are filled
    with 0 (shouldn't normally happen if engineer_features is correct,
    but this is a safety net).
    """
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]