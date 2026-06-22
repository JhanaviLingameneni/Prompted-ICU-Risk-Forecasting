"""
main.py - Preprocess PhysioNet 2012 ICU data from all sets

Usage:
    python main.py --data_dirs ../data/set-a ../data/set-b ../data/set-c --outcomes ../data/Outcomes-a.txt ../data/Outcomes-b.txt ../data/Outcomes-c.txt --output_dir ../results
"""

import argparse
import os
from data_preprocessing import build_dataset, clean_dataset


def main():
    parser = argparse.ArgumentParser(description="ICU Mortality - Data Preprocessing")
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--outcomes", nargs="+", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Building dataset from raw patient files...")
    print("=" * 60)
    df = build_dataset(args.data_dirs, args.outcomes)

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning and feature engineering...")
    print("=" * 60)
    df_clean, feature_cols, target_col = clean_dataset(df)

    output_path = os.path.join(args.output_dir, "processed.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    print(f"Shape: {df_clean.shape[0]} patients, {len(feature_cols)} features")
    print("Done!")


if __name__ == "__main__":
    main()