"""
data_preprocessing.py - Preprocess PhysioNet 2012 ICU patient data

Reads raw patient .txt files from set-a, set-b, and set-c,
extracts features, cleans data, and outputs a flat CSV ready for modeling.

Usage:
    python data_preprocessing.py --data_dirs ../data/set-a ../data/set-b ../data/set-c --outcomes ../data/Outcomes-a.txt ../data/Outcomes-b.txt ../data/Outcomes-c.txt --output ../results/processed.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

STATIC_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}

TS_PARAMS = [
    "HR", "SysABP", "DiasABP", "MAP", "Temp", "RespRate", "GCS",
    "Glucose", "BUN", "Creatinine", "HCT", "WBC", "Platelets",
    "Na", "K", "pH", "Urine", "MechVent",
]

STATS = ["mean", "min", "max", "std"]
SENTINEL = -1


def parse_patient_file(filepath):
    static = {}
    ts_records = []

    with open(filepath, "r") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_str, param, value = line.split(",")
            value = float(value)

            if param in STATIC_PARAMS:
                static[param] = value
            else:
                ts_records.append({"parameter": param, "value": value})

    ts_df = pd.DataFrame(ts_records) if ts_records else pd.DataFrame(columns=["parameter", "value"])
    return static, ts_df


def extract_ts_features(ts_df):
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
        features[f"{param}_std"] = np.nanstd(values) if len(values) > 1 else 0.0

    return features


def build_dataset(data_dirs, outcomes_paths):
    outcomes_list = []
    for path in outcomes_paths:
        df = pd.read_csv(path)
        df.rename(columns=lambda c: c.strip(), inplace=True)
        outcomes_list.append(df)
    outcomes = pd.concat(outcomes_list, ignore_index=True)
    print(f"Loaded {len(outcomes)} total outcomes from {len(outcomes_paths)} files")

    all_files = []
    for data_dir in data_dirs:
        files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(".txt") and not f.startswith("Outcomes")
        ])
        all_files.extend(files)
        print(f"  {data_dir}: {len(files)} patient files")

    print(f"Total patient files: {len(all_files)}")

    all_rows = []
    for i, filepath in enumerate(all_files):
        static, ts_df = parse_patient_file(filepath)

        row = {
            "RecordID": int(static.get("RecordID", 0)),
            "Age": static.get("Age", np.nan),
            "Gender": static.get("Gender", np.nan),
            "Height": static.get("Height", np.nan),
            "Weight_static": static.get("Weight", np.nan),
            "ICUType": static.get("ICUType", np.nan),
        }
        row.update(extract_ts_features(ts_df))
        all_rows.append(row)

        if (i + 1) % 1000 == 0 or (i + 1) == len(all_files):
            print(f"  Processed {i + 1}/{len(all_files)} patients...")

    df = pd.DataFrame(all_rows)
    df = df.merge(outcomes, on="RecordID", how="inner")
    print(f"Final dataset shape: {df.shape}")
    return df


def clean_dataset(df):
    target_col = "In-hospital_death"

    for col in ["Height", "Weight_static"]:
        if col in df.columns:
            df[col] = df[col].replace(SENTINEL, np.nan)

    range_checks = {
        "Age": (0, 120), "Height": (50, 250), "Weight_static": (20, 300),
        "HR_mean": (10, 300), "Temp_mean": (25, 45), "SysABP_mean": (20, 300),
        "DiasABP_mean": (10, 200), "Glucose_mean": (10, 1500),
        "pH_mean": (6.5, 8.0), "GCS_min": (3, 15),
    }
    for col, (lo, hi) in range_checks.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Feature engineering
    height_m = df["Height"] / 100.0
    df["BMI"] = df["Weight_static"] / (height_m ** 2)
    df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan

    df["ShockIndex"] = df["HR_mean"] / df["SysABP_mean"].replace(0, np.nan)
    df["PulsePressure"] = df["SysABP_mean"] - df["DiasABP_mean"]
    df["BUN_Creatinine_Ratio"] = df["BUN_mean"] / df["Creatinine_mean"].replace(0, np.nan)

    # One-hot encode ICU type
    icu_dummies = pd.get_dummies(df["ICUType"], prefix="ICU", dtype=int)
    df = pd.concat([df, icu_dummies], axis=1)
    df.drop("ICUType", axis=1, inplace=True)

    # Drop non-feature columns
    drop_cols = [c for c in ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival"]
                 if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Drop features with >80% missing
    feature_cols = [c for c in df.columns if c != target_col]
    missing_pct = df[feature_cols].isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 80].index.tolist()
    if high_missing:
        print(f"  Dropping {len(high_missing)} features with >80% missing: {high_missing}...")
        df.drop(columns=high_missing, inplace=True)
        feature_cols = [c for c in feature_cols if c not in high_missing]

    # Fill missing values with median
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # Move target column to the end
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]

    print(f"  After cleaning: {df.shape[0]} patients, {len(feature_cols)} features")
    print(f"  Class distribution:\n{df[target_col].value_counts().to_string()}")
    return df, feature_cols, target_col


def main():
    parser = argparse.ArgumentParser(description="Preprocess PhysioNet 2012 data")
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--outcomes", nargs="+", required=True)
    parser.add_argument("--output", default="processed.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Building dataset from raw patient files...")
    print("=" * 60)
    df = build_dataset(args.data_dirs, args.outcomes)

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