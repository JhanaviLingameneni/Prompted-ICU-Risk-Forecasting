"""
This module contains functions for loading and preprocessing the
patient data and outcomes from the PhysioNet Challenge 2012 dataset.
@see https://physionet.org/content/challenge-2012/1.0.0/
"""

import glob
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# As per dataset description, these are the general descriptors that are not vitals.
GENERAL_DESCRIPTORS = { "Time", "RecordID", "Age", "Gender", "Height", "ICUType", "Weight" }
BIOMETRICS = GENERAL_DESCRIPTORS - {"Time", "ICUType"}


def clean_patient_data(patient_file: str) -> pd.DataFrame:
    """
    Cleans a single patient's data.

    Args:
        patient_file (str): The path to the patient's data file.

    Returns:
        pd.DataFrame: The cleaned patient time series with columns:
        RecordID, Time, and one column per vital sign.
    """
    df = pd.read_csv(patient_file)

    # Clean time into minutes since.
    def to_minutes(time_str):
        h, m = map(int, time_str.split(':'))
        return h * 60 + m

    df["Time"] = df["Time"].apply(to_minutes)

    # Extract RecordID and remove it. We'll convert it into its own column.
    recordid = df.loc[df["Parameter"] == "RecordID", "Value"].values[0]
    df = df[df["Parameter"] != "RecordID"]

    # Pivot the data to have parameters as columns and timestamps as rows
    df = df.pivot_table(index="Time", columns="Parameter",
                        values="Value")

    # Sort by time for aggregation later.
    df.sort_index(inplace=True)

    # Per dataset description. -1 means it is missing value
    df = df.mask(df == -1)

    # Flatten to a tidy table for concatenation across all patients.
    df = df.reset_index()
    df["RecordID"] = int(recordid)
    return df

def process_dataset(data_set: str, undersample: bool = False):
    """
    Loads and processes the patient data from the specified directory.
    Each set is expected to reside in a directory named
    "set-a", "set-b", or "set-c" under the data directory.

    Args:
        undersample: Whether to apply random undersampling on the majority
            class. This should generally be enabled only for training data.

    Returns:
        (pd.DataFrame, pd.DataFrame): A tuple containing the processed
        patient data and the corresponding outcomes.
        The shape of features is (record_id, vitals..., aggregated_vitals..., survival),
        where each time-varying vital is expanded into four features:
        *_mean, *_median, *_min, *_max, and *_std.
        Corresponding outcomes is a DataFrame indexed by RecordID containing Survival.
            -1 indicates patient survived.
    """
    if data_set not in {"a", "b", "c"}:
        raise ValueError('data_set must be one of "a", "b", or "c"')

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    dir_path = os.path.join(data_root, f"set-{data_set}")
    fp = glob.glob(os.path.join(dir_path, "*.txt"))
    patient_data = []

    for f in fp:
        patient_data.append(clean_patient_data(f))

    df_all = pd.concat(patient_data, ignore_index=True)

    # Grab all vital columns. Exclude general descriptors.
    vital_columns = [c for c in df_all.columns if c not in GENERAL_DESCRIPTORS]

    # Grab biometric columns, these should not be aggregated
    biometric_columns = [c for c in df_all.columns if c in BIOMETRICS]

    # Aggregate each vital over time into one row per patient.
    grouped = df_all.groupby("RecordID")[vital_columns]
    df_mean = grouped.mean().add_suffix("_mean")
    df_median = grouped.median().add_suffix("_median")
    df_min = grouped.min().add_suffix("_min")
    df_max = grouped.max().add_suffix("_max")
    df_std = grouped.std(ddof=0).add_suffix("_std")
    df_first = grouped.first().add_suffix("_first")
    df_last = grouped.last().add_suffix("_last")
    df_count = grouped.count().add_suffix("_count")
    df_missing = grouped.count().eq(0).astype(int).add_suffix("_missing")

    df_delta = (
        df_last.rename(columns=lambda c: c.replace("_last", "")) -
        df_first.rename(columns=lambda c: c.replace("_first", ""))
    ).add_suffix("_delta")

    df_range = (
        df_max.rename(columns=lambda c: c.replace("_max", "")) -
        df_min.rename(columns=lambda c: c.replace("_min", ""))
    ).add_suffix("_range")

    df_features = pd.concat(
        [
            df_mean, df_median, df_min, df_max, df_std,
            df_first, df_last, df_delta, df_missing, df_range
        ],
        axis=1
    )

    # Fill remaining NaNs
    df_features = df_features.fillna(df_features.median(numeric_only=True))

    # Fill missing biometrics: median for continuous, mode for Gender.
    # Would love to drop it but too many records are missing it.
    continuous_biometrics = [
        c for c in biometric_columns if c not in {"RecordID", "Gender"}]
    bio_medians = df_all[continuous_biometrics].median()
    df_all[continuous_biometrics] = df_all[continuous_biometrics].fillna(
        bio_medians)
    gender_mode = df_all["Gender"].mode()[0]
    df_all["Gender"] = df_all["Gender"].fillna(gender_mode)

    # One-hot encode ICUType
    icu_dummies = pd.get_dummies(df_all[["RecordID", "ICUType"]], columns=[
                                 "ICUType"], dummy_na=True)
    icu_dummies = icu_dummies.drop_duplicates(
        subset="RecordID").set_index("RecordID").astype(float)
    df_features = df_features.join(icu_dummies)

    # Add the biometrics, unaggregated.
    df_features = df_features.join(df_all[biometric_columns].drop_duplicates(
        subset="RecordID").set_index("RecordID"), how="left")

    # Sort columns to guarantee consistent feature order across datasets.
    df_features = df_features[sorted(df_features.columns)]

    outcomes = _load_outcomes(data_set, data_root)

    if undersample:
        rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        return rus.fit_resample(df_features, outcomes)

    return df_features, outcomes

def _load_outcomes(data_set: str, data_root: str) -> pd.DataFrame:
    """
    Load survival outcomes for the selected dataset.
    -1 survival indicates the patient survived. All other numbers indicate days until death.

    Reads the outcomes file that corresponds to this loader's dataset
    identifier and returns only the columns needed for supervised learning.

    Columns:
        RecordID: Unique identifier for each patient record.
        Survival: The number of days between ICU admission and death or -1
                    if the patient survived. This excludes patients who
                    spent less than 48 hours in the ICU.

    Returns:
        pd.DataFrame: A DataFrame indexed by RecordID containing Survival.
                        Death is a binary column where 1 indicates risk and 0 indicates
                        no immediate risk.
    """
    outcomes_file_path = os.path.join(data_root, "outcomes", f"outcomes-{data_set}.txt")

    # We only care about RecordID and Survival
    df = pd.read_csv(outcomes_file_path, usecols=["RecordID", "Survival", "In-hospital_death"])
    df["Death"] = (df["In-hospital_death"] == 1).astype(int)
    df = df.drop(columns=["Survival", "In-hospital_death"])
    return df.set_index("RecordID")
