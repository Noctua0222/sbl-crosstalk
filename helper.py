# Crosstalk Project

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import crosstalk_main

def get_train_test_split_small() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
    """
    This function performs the following steps:
    - Reads in the data
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population
    
    After all these steps, it splits the data into 80% train and 20% test. 
    
    The labels are continuous, taken from the correlation value
    - Labels from cor 

    Features:
    - ac_loc
    - pho_loc
    
    Returns the features and labels for train and test sets, followed by the names of features.
    """
    path =''
    df_cor_n = pd.read_csv(path+'data/correlation_n.csv')
    df_cor_n = df_cor_n[['ac_loc','pho_loc', 'cor']] # Select desired columns
    df_cor_n = df_cor_n[:2000]

    crosstalks = {}
    for i in tqdm(range(2000), desc='Loading files from disk'):
        crosstalks[i] = df_cor_n[i:i+1]
    
    features = Parallel(n_jobs=16)(delayed(crosstalk_main.generate_feature_vector)(df) for _, df in tqdm(crosstalks.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_cor_n['cor'].values
    X = crosstalk_main.impute_missing_values(X)
    X = crosstalk_main.normalize_feature_matrix(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names