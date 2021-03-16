import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def feature_target_split(df, target):
    """
    Splits DataFrame into features and target df
    
    Inputs:
        df (df): The initial DataFrame
        target (str): the target variable
        
    Returns:
        X and y DataFrames
    """
    X = df.drop(columns=target)
    y = df["target"]
    
    return X, y


def cut_outliers(data, col_name):
    """
    Filter out the outliers in a column of the data.
    Inputs:
        data: (DataFrame) the full data
        col_name: (string) the name of the column in need of cut outliers
    Returns:
        data: (DataFrame) the updated full data
    """
    col = data[col_name]
    data = data[np.abs(col-col.mean()) <= (3 * col.std())]
    return data


def discretize(data, bins, list_to_discrete):
    """
    Discretize the value of certain columns (specified in list_to_discrete)
    in the full data. Note that this function transforms data in place.
    Inputs:
        data: (DataFrame) the full data
        bins: (int) the bins to discretize the column. For example, if bins
            is 3, then the values of the data are cut into 3 ranges in order,
            and are transformed to 0, 1 and 2 according to its original value.
        list_to_discrete: (list) the list storing the name of columns to be
            discretized.
    Returns:
        None
    """
    for col_name in list_to_discrete:
        data[col_name] = pd.cut(data[col_name], bins, right=True,
                                precision=4).cat.codes


def get_dummies(X):
    """
    Get dummy variables for categorical features
    
    Inputs:
        X (df): the feature DataFrame to get dummy variables for
        
    Returns:
        Feature DataFrame with dummy variables
    """
    return pd.get_dummies(df)


def split(features, target, test_size=0.2, random_state=0):
    """
    Splits data into training and testing set.
    
    Inputs:
        features (df): the feature columns of the data
        target (df): the target column of the data
        test_size (float): the proportion of of test to training data
        random_state (int): the random state to be chosen
        
    Returns:
        Four DataFrame objects
        
    example:
        X_train, X_test, y_train, y_test = split(features, target)
    """
    
    return train_test_split(features, target)


def scale(X_train, X_test, scale_type = "minmax"):
    """
    Scales Feature Data.
    
    Inputs:
        X_train (df): the feature data used for training
        X_test (df): the feature data used for testing
        type (str): whether the data should be scaled via MinMax or Standard
    
    Returns:
        Standardized X_train and X_test.
    """
    if scale_type == "mixmax":
        scaler = MinMaxScaler()
    if scale_type == "standard":
        scaler = StandardScaler()
        
    return scaler.fit_transform(X_train), scaler.transform(X_test)