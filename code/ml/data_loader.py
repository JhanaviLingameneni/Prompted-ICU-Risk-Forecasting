"""
This module contains the DataLoader class, which is responsible for loading and preprocessing the
patient data and outcomes from the PhysioNet Challenge 2012 dataset.
@see https://physionet.org/content/challenge-2012/1.0.0/
"""

import glob
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    DataLoader is responsible for loading and processing the patient data and outcomes.
    """

    # As per dataset description, these are the general descriptors that are not vitals.
    GENERAL_DESCRIPTORS = { "Time", "RecordID", "Age", "Gender", "Height", "ICUType", "Weight" }
    BIOMETRICS = GENERAL_DESCRIPTORS - { "Time" }

    def __init__(self, data_set: str):
        """
        Initializes the DataLoader with the specified dataset identifier.
        Args:
            data_set (str): The identifier for the dataset to load. One of "a", "b" or "c".
        """
        if data_set not in {"a", "b", "c"}:
            raise ValueError('data_set must be one of "a", "b", or "c"')

        self.data_set = data_set
        self._data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


    def clean_patient_data(self, patient_file: str) -> pd.DataFrame:
        """
        Cleans a single patients data.

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
        df = df[df["Parameter"] != "ICUType"]

        # Pivot the data to have parameters as columns and timestamps as rows
        df = df.pivot_table(index="Time", columns="Parameter", values="Value", aggfunc='mean')

        # Per dataset description. -1 means it is missing value
        df = df.mask(df == -1)
        patient_means = df.mean(axis=0, skipna=True)
        df = df.fillna(patient_means)

        # Flatten to a tidy table for concatenation across all patients.
        df = df.reset_index()
        df["RecordID"] = int(recordid)
        return df

    def scale_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the dataset using StandardScaler.

        Args:
            df (pd.DataFrame): The input DataFrame to scale.
        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
        return scaled_df


    def process_dataset(self) -> pd.DataFrame:
        """
        Loads and processes the patient data from the specified directory.
        Each set is expected to reside in a directory named
        "set-a", "set-b", or "set-c" under the data directory.

        Returns:
            (pd.DataFrame, pd.DataFrame): A tuple containing the processed
            patient data and the corresponding outcomes.
            The shape of features is (record_id, vitals..., aggregated_vitals..., survival),
            where each time-varying vital is expanded into four features:
            *_mean, *_median, *_min, and *_max.
            Corresponding outcomes is a DataFrame indexed by RecordID containing Survival.
             -1 indicates patient survived.
        """
        dir_path = os.path.join(self._data_root, f"set-{self.data_set}")
        fp = glob.glob(os.path.join(dir_path, "*.txt"))
        patient_data = []

        for f in fp:
            patient_data.append(self.clean_patient_data(f))

        df_all = pd.concat(patient_data, ignore_index=True)

        # Grab all vital columns. Exclude general descriptors.
        vital_columns = [c for c in df_all.columns if c not in self.GENERAL_DESCRIPTORS]
        # Grab biometric columns, these should not be aggregated
        biometric_columns = [c for c in df_all.columns if c in self.BIOMETRICS]

        # Fill any remaining missing values using global medians for each vital.
        global_medians = df_all[vital_columns].median(axis=0, skipna=True)
        df_all[vital_columns] = df_all[vital_columns].fillna(global_medians)

        # Aggregate each vital over time into one row per patient.
        grouped = df_all.groupby("RecordID")[vital_columns]
        df_mean = grouped.mean().add_suffix("_mean")
        df_median = grouped.median().add_suffix("_median")
        df_min = grouped.min().add_suffix("_min")
        df_max = grouped.max().add_suffix("_max")

        df_features = pd.concat([df_mean, df_median, df_min, df_max], axis=1)

        # Add the biometrics, unaggregated.
        df_features = df_features.join(df_all[biometric_columns].drop_duplicates(subset="RecordID").set_index("RecordID"), how="left")
        df_features = self.scale_dataset(df_features)

        return df_features, self._load_outcomes()

    def _load_outcomes(self) -> pd.DataFrame:
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
        """
        outcomes_file_path = os.path.join(self._data_root, "outcomes", f"outcomes-{self.data_set}.txt")
        # We only care about RecordID and Survival
        df = pd.read_csv(outcomes_file_path, usecols=["RecordID", "Survival"])
        return df.set_index("RecordID")
