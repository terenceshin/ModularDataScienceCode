import pandas as pd
import xgboost as xgb


def xgb_classifier(X_train, y_train, params):
    """
    Creates XGBoost Classifier
    
    Inputs:
        X_train (df): the feature training data
        y_train (df): the target training data
        params (dict): the parameters of XGBoost
        
    Returns:
        XGB Classifier
    """
    model = xgb.XGBClassifier(**params)
    return model.fit(X_train, y_train)


def make_predictions(model, X_test):
    """
    Makes predictions on validate/test data
    
    Inputs:
        model (object): the model to be used for the new data
        X_test (df): the data to feed through the model
        
    Returns:
        Model predictions
    """  
    return model.predict(X_test)


def plot_feat_importance(model, feature_names, size=10):
    """
    plots feature importance of a model
    
    Inputs:
        model (object): the model which its features will be plotted
        feature_names (list): the list of feature names
        size (int): the number of features to plot 
        
    Returns:
        Plot of feature importance for a given model
    """  
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feature_importances.nlargest(size).plot(kind='barh', figsize=(20,20))


def feat_imp_to_df(model, feature_names):
    """
    Writes feature importance to a DataFrame
    
    Inputs:
        model (object): the model that we want the feature importance
        feature_names (list): the list of feature names
        
    Returns:
        DataFrame of feature importance
    """  
    df = pd.DataFrame(model.feature_importances_, feature_names) \
           . reset_index() \
           .rename(columns={'index': 'feature', 0: 'feat_importance'}) \
           .sort_values(by='feat_importance', ascending=False)
    
    return df


