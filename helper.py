# Crosstalk Project

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

def get_train_test_split() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
    """
    This function performs the following steps:
    - Reads in the data from data/labels.csv and data/files/*.csv (keep only the first 2,500 examples)
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population
    
    After all these steps, it splits the data into 80% train and 20% test. 
    
    The binary labels take two values:
        -1: survivor
        +1: died in hospital
    
    Returns the features and labels for train and test sets, followed by the names of features.
    """
    path =''
    df_labels = pd.read_csv(path+'data/labels.csv')
    df_labels = df_labels[:2000]
    IDs = df_labels['RecordID'][:2000]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv(f'{path}data/files/{i}.csv')
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values
    X = crosstalk_main.impute_missing_values(X)
    X = crosstalk_main.normalize_feature_matrix(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names