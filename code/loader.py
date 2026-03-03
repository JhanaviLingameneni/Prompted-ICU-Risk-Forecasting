# https://physionet.org/content/challenge-2012/1.0.0/

import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    
    return df

def load_data(file_path: str) -> pd.DataFrame:
    chunksize == 10000
    chunks = []
    with pd.read_csv(file_path, chunksize=chunksize) as reader:
        for chunk in reader:
            chunks.append(preprocess_data(chunk))
    return pd.concat(chunks, ignore_index=True)