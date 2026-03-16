"""
data_preprocessing.py
=====================
Loads raw PhysioNet Challenge 2012 patient time-series files,
performs cleaning, feature engineering, and outputs a flat
one-row-per-patient DataFrame ready for modeling.

Usage:
    python data_preprocessing.py --data_dir ../data/set-a --outcomes ../data/Outcomes-a.txt --output ../data/processed.csv
"""

import os
import argparse
import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

# Static (recorded once at time 00:00) vs. time-series parameters
STATIC_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}

# Only the most clinically important time-series parameters (18 selected)
TS_PARAMS = [
    "HR",           # Heart rate — basic vital sign
    "SysABP",       # Systolic blood pressure
    "DiasABP",      # Diastolic blood pressure
    "MAP",          # Mean arterial pressure
    "Temp",         # Temperature
    "RespRate",     # Respiratory rate
    "GCS",          # Glasgow Coma Scale (consciousness)
    "Glucose",      # Blood sugar
    "BUN",          # Blood urea nitrogen (kidney function)
    "Creatinine",   # Kidney function
    "HCT",          # Hematocrit (blood cells %)
    "WBC",          # White blood cells (infection marker)
    "Platelets",    # Clotting ability
    "Na",           # Sodium
    "K",            # Potassium
    "pH",           # Blood acidity
    "Urine",        # Urine output (kidney function)
    "MechVent",     # On mechanical ventilator (0 or 1)
]

# We compute only 3 stats per parameter: mean, min, max
STATS = ["mean", "min", "max"]

# Sentinel values used for missing data in the raw files
SENTINEL = -1


# ── Helper Functions ─────────────────────────────────────────────────────────

def parse_patient_file(filepath):
    """
    Parse a single patient .txt file into static features and
    a time-indexed DataFrame of time-series measurements.

    :param filepath: Path to the patient .txt file.
    :return: Tuple of (static_dict, ts_dataframe).
    """
    static = {}
    ts_records = []

    with open(filepath, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_str, param, value = line.split(",")
            value = float(value)

            if param in STATIC_PARAMS:
                static[param] = value
            else:
                ts_records.append({
                    "parameter": param,
                    "value": value
                })

    ts_df = pd.DataFrame(ts_records) if ts_records else pd.DataFrame(
        columns=["parameter", "value"]
    )
    return static, ts_df


def extract_ts_features(ts_df):
    """
    From a patient's time-series DataFrame, compute mean, min, max
    for each important parameter.

    :param ts_df: DataFrame with columns [parameter, value].
    :return: Dict of feature_name -> value.
    """
    features = {}

    for param in TS_PARAMS:
        subset = ts_df[ts_df["parameter"] == param]

        if subset.empty:
            for stat in STATS:
                features[f"{param}_{stat}"] = np.nan
            continue

        values = subset["value"].values
        features[f"{param}_mean"] = np.nanmean(values)
        features[f"{param}_min"] = np.nanmin(values)
        features[f"{param}_max"] = np.nanmax(values)

    return features


def build_dataset(data_dir, outcomes_path):
    """
    Process all patient files and merge with outcomes.

    :param data_dir: Directory containing patient .txt files.
    :param outcomes_path: Path to the Outcomes-a.txt file.
    :return: pandas DataFrame with one row per patient.
    """
    # Load outcomes
    outcomes = pd.read_csv(outcomes_path)
    outcomes.rename(columns=lambda c: c.strip(), inplace=True)

    # Get list of patient files
    patient_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".txt") and f != "Outcomes-a.txt"
    ])

    print(f"Found {len(patient_files)} patient files in {data_dir}")

    all_rows = []
    for i, fname in enumerate(patient_files):
        filepath = os.path.join(data_dir, fname)
        static, ts_df = parse_patient_file(filepath)

        # Start with static features
        row = {}
        row["RecordID"] = int(static.get("RecordID", 0))
        row["Age"] = static.get("Age", np.nan)
        row["Gender"] = static.get("Gender", np.nan)
        row["Height"] = static.get("Height", np.nan)
        row["Weight_static"] = static.get("Weight", np.nan)
        row["ICUType"] = static.get("ICUType", np.nan)

        # Extract time-series features (mean, min, max only)
        ts_feats = extract_ts_features(ts_df)
        row.update(ts_feats)

        all_rows.append(row)

        if (i + 1) % 500 == 0 or (i + 1) == len(patient_files):
            print(f"  Processed {i + 1}/{len(patient_files)} patients...")

    df = pd.DataFrame(all_rows)

    # Merge with outcomes
    df = df.merge(outcomes, on="RecordID", how="inner")

    print(f"Final dataset shape: {df.shape}")
    return df


def clean_dataset(df):
    """
    Clean the merged dataset:
      - Replace sentinel values (-1) with NaN
      - Handle impossible values
      - Engineer new features
      - One-hot encode ICUType
      - Impute missing values with median

    :param df: Raw merged DataFrame.
    :return: Tuple of (cleaned_df, feature_columns, target_column).
    """
    target_col = "In-hospital_death"

    # ── Step 1: Replace sentinel values ──────────────────────────────────
    for col in ["Height", "Weight_static"]:
        if col in df.columns:
            df[col] = df[col].replace(SENTINEL, np.nan)

    # ── Step 2: Handle impossible / out-of-range values ──────────────────
    range_checks = {
        "Age": (0, 120),
        "Height": (50, 250),
        "Weight_static": (20, 300),
        "HR_mean": (10, 300),
        "Temp_mean": (25, 45),
        "SysABP_mean": (20, 300),
        "DiasABP_mean": (10, 200),
        "Glucose_mean": (10, 1500),
        "pH_mean": (6.5, 8.0),
        "GCS_min": (3, 15),
    }
    for col, (lo, hi) in range_checks.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # ── Step 3: Feature engineering ──────────────────────────────────────
    # BMI
    if "Height" in df.columns and "Weight_static" in df.columns:
        height_m = df["Height"] / 100.0
        df["BMI"] = df["Weight_static"] / (height_m ** 2)
        df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan

    # Shock index (HR / SysABP) — values > 0.7 suggest shock
    if "HR_mean" in df.columns and "SysABP_mean" in df.columns:
        df["ShockIndex"] = df["HR_mean"] / df["SysABP_mean"].replace(0, np.nan)

    # Pulse pressure
    if "SysABP_mean" in df.columns and "DiasABP_mean" in df.columns:
        df["PulsePressure"] = df["SysABP_mean"] - df["DiasABP_mean"]

    # BUN/Creatinine ratio (kidney function)
    if "BUN_mean" in df.columns and "Creatinine_mean" in df.columns:
        df["BUN_Creatinine_Ratio"] = df["BUN_mean"] / df["Creatinine_mean"].replace(0, np.nan)

    # ── Step 4: One-hot encode ICUType ───────────────────────────────────
    if "ICUType" in df.columns:
        icu_dummies = pd.get_dummies(df["ICUType"], prefix="ICU", dtype=int)
        df = pd.concat([df, icu_dummies], axis=1)
        df.drop("ICUType", axis=1, inplace=True)

    # ── Step 5: Drop non-feature columns ─────────────────────────────────
    drop_cols = ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ── Step 6: Identify feature columns ─────────────────────────────────
    feature_cols = [c for c in df.columns if c != target_col]

    # ── Step 7: Drop features with >80% missing ─────────────────────────
    missing_pct = df[feature_cols].isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 80].index.tolist()
    if high_missing:
        print(f"  Dropping {len(high_missing)} features with >80% missing: {high_missing}...")
        df.drop(columns=high_missing, inplace=True)
        feature_cols = [c for c in feature_cols if c not in high_missing]

    # ── Step 8: Impute missing values with median ────────────────────────
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    print(f"  After cleaning: {df.shape[0]} patients, {len(feature_cols)} features")
    print(f"  Class distribution:\n{df[target_col].value_counts().to_string()}")

    return df, feature_cols, target_col


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess PhysioNet 2012 data")
    parser.add_argument("--data_dir", required=True, help="Directory with patient .txt files")
    parser.add_argument("--outcomes", required=True, help="Path to Outcomes-a.txt")
    parser.add_argument("--output", default="processed.csv", help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Building dataset from raw patient files...")
    print("=" * 60)
    df = build_dataset(args.data_dir, args.outcomes)

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning and feature engineering...")
    print("=" * 60)
    df_clean, feature_cols, target_col = clean_dataset(df)

    print(f"\nSaving processed data to {args.output}...")
    df_clean.to_csv(args.output, index=False)
    print("Done!")

    return df_clean, feature_cols, target_col


if __name__ == "__main__":
    main()