"""
This module contains the DataLoader class, which is responsible for loading and preprocessing the
patient data and outcomes from the PhysioNet Challenge 2012 dataset.
@see https://physionet.org/content/challenge-2012/1.0.0/
"""

import glob
import os
import pandas as pd
import xarray as xr


class DataLoader:
    """
    DataLoader is responsible for loading and processing the patient data and outcomes.
    """

    def __init__(self, data_set: str):
        """
        Initializes the DataLoader with the specified dataset identifier.
        Args:
            data_set (str): The identifier for the dataset to load. One of "a", "b" or "c".
        """
        if data_set not in {"a", "b", "c"}:
            raise ValueError('data_set must be one of "a", "b", or "c"')

        self.data_set = data_set
        # Outcomes with RecordID as indexed DataArray for easy merging with patient data
        self.outcomes = self._load_outcomes()


    def clean_patient_data(self, patient_file: str) -> xr.Dataset:
        """
        Cleans a single patients data.

        Args:
            patient_file (str): The path to the patient's data file.

        Returns:
            xr.Dataset: The cleaned dataset.
        """
        df = pd.read_csv(patient_file)

        # Extract RecordID and remove it. We'll convert it into its own column.
        recordid = df.loc[df["Parameter"] == "RecordID", "Value"].values[0]
        df = df[df["Parameter"] != "RecordID"]

        # Pivot the data to have parameters as columns and timestamps as rows
        df = df.pivot(index="Time", columns="Parameter", values="Value")

        # Convert to DataArray and add RecordID as a new dimension
        ds = df.to_xarray().expand_dims(RecordID=[int(recordid)])

        # Remove records with missing values
        ds = ds.where(ds != -1)

        # Fill missing values with the mean of that patient
        patient_means = ds.mean(dim="Time", skipna=True)

        return ds.fillna(patient_means)



    def process_dataset(self) -> pd.DataFrame:
        """
        Loads and processes the patient data from the specified directory.
        Each set is expected to reside in a directory named "set-a", "set-b", or "set-c" under the data directory.

        Returns:
            pd.DataFrame: A DataFrame containing the processed patient data.
        """
        dir_path = os.path.join("..", "data", f"set-{self.data_set}")
        fp = glob.glob(os.path.join(dir_path, "*.txt"))
        patient_data = []

        for f in fp:
            print(f"Loading {f}...")
            patient_data.append(self.clean_patient_data(f))

        ds_all = xr.concat(patient_data, dim="RecordID")

        # Clean rest of missing data not found otherwise in timeseries by filling
        # with global median across all patients.
        global_means = ds_all.median(dim=["RecordID", "Time"], skipna=True)
        ds_all = ds_all.fillna(global_means)

        ds_all = ds_all.merge(self.outcomes)

        # Clean the data and convert back to DataFrame for easier downstream work with scikit-learn and pandas.
        return ds_all.to_dataframe()

    def _load_outcomes(self) -> xr.DataArray:
        """
        Load survival outcomes for the selected dataset.

        Reads the outcomes file that corresponds to this loader's dataset
        identifier and returns only the columns needed for supervised learning.

        Columns:
            RecordID: Unique identifier for each patient record.
            Survival: The number of days between ICU admission and death or -1
                      if the patient survived. This excludes patients who
                      spent less than 48 hours in the ICU.

        Returns:
            xr.DataArray: A DataArray containing RecordID and Survival.
        """
        outcomes_file_path = os.path.join("..", "data", "outcomes", f"outcomes-{self.data_set}.txt")
        # We only care about RecordID and Survival
        df = pd.read_csv(outcomes_file_path, usecols=["RecordID", "Survival"])
        return df.set_index("RecordID").to_xarray()
