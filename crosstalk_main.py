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

# Helper Functions ================================================= #
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    return None

def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return None

def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return None

# main() =========================================================== #

def main() -> None:
    # Driver

    return None

if __name__ == "__main__":
    main()