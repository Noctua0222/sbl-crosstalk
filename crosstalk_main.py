# Crosstalk Project - Main file

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)

# Helper Functions ================================================= #
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing a list of crosstalks

    Args:
        df: dataframe with columns [ ac_loc, pho_loc, cor ]

    Returns:
        a dictionary of format {feature_name: feature_value}
        for example, {'ac_loc': 324, 'pho_loc': 329}
    """
    variables = config["variables"]

    feature_dict = {}
    # Add values into dictionary
    for i, row in df.iterrows():
        for variable in variables:
            value = row[variable]
            feature_dict[variable] = value.item()

    return feature_dict

def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return X

def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Normalize scale of features
    scaler = MinMaxScaler(feature_range=(0, 1)) # default is feature_range=(0, 1)
    model = scaler.fit(X)

    # The transform() method allows you to execute a function for each value of the DataFrame.
    normalized_X = model.transform(X) 

    return normalized_X

def remove_unlabeled_data(df: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return None

def binarize_values(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    y = np.where(y >= 0 , np.ones_like(y), np.full_like(y, -1))

    return y

# main() =========================================================== #

def main() -> None:
    # Driver

    # Read data -> first 2000 crosstalks
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split_small()

    return None

if __name__ == "__main__":
    main()