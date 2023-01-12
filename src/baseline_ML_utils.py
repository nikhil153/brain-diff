import numpy as np
import pandas as pd
from datetime import datetime

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def concat_sessions(df, config, features):
    """ Concats baseline and followup sessions if present"""
    X_list = []
    y_list = []
    labels = config["labels"]
    for ses in config["sessions"]:        
        X = df[df["session"]==ses][features].values                
        X_list.append(X)

    if len(X_list) > 1:
        X_train = np.concatenate(X_list, axis=1)
    else:
        X_train = np.array(X_list)
    
    y_train = np.squeeze(df[df["session"]==ses][labels].values)

    return np.squeeze(X_train), np.squeeze(y_train)

def get_ML_data(exp_data_df, exp_config, testdata_only=False):
    """ Generates train and / or test arrays for a given exp_config and and exp_data_df"""
    features = exp_config["features"]
    train_data_dict = {}
    if not testdata_only: 
        train_config = exp_config["train_config"]
        # Generate train arrays
        train_df = exp_data_df[(exp_data_df["CV_subset"] == train_config["subset"]) & 
                                (exp_data_df["session"].isin(train_config["sessions"]))].copy()

        X_train, y_train = concat_sessions(train_df, train_config, features)

        train_data_dict["train"] = (X_train, y_train)
    
    # Generate test arrays    
    test_data_dict = {}
    test_configs = exp_config["test_configs"]
    for name, test_config in test_configs.items():
        test_df = exp_data_df[(exp_data_df["CV_subset"] == test_config["subset"]) & 
                            (exp_data_df["session"].isin(test_config["sessions"]))].copy()

        X_test, y_test = concat_sessions(test_df, test_config, features)

        test_data_dict[name] = (X_test, y_test)

    return train_data_dict, test_data_dict

def calculate_perf_metrics(y_pred, y_test, visit="BL"):
    """ calculates perf metrics """
    sq_err = (y_pred - y_test)**2
    abs_err = np.abs(y_pred - y_test)
    corr = stats.pearsonr(y_pred, y_test)[0]

    df = pd.DataFrame()
    df[f"y_test_{visit}"] = y_test # First visit perf
    df[f"y_pred_{visit}"] = y_pred
    df[f"sq_err_{visit}"] = sq_err
    df[f"abs_err_{visit}"] = abs_err
    df[f"corr_{visit}"] = corr

    return df
    
def get_brain_age_perf(model, X_CV, y_CV, X_test, y_test, X_test_FU=None, y_test_FU=None, pretrained=False, cv=2):
    """ Compute CV score and heldout sample MAE and correlation. 
        This is used with baseline sklearn models.
    """
    y_test = np.squeeze(y_test)

    if pretrained:
        print(f"***Using pretrained model***")
        pipeline = model

    else:
        print(f"***Training a new model***")
        y_CV = np.squeeze(y_CV)
        y_CV = y_CV/100 #scale age

        pipeline = Pipeline([("brainage_model", model)])
        pipeline.fit(X_CV, y_CV)

        # Evaluate the models using crossvalidation
        CV_scores = cross_val_score(pipeline, X_CV, y_CV,scoring="neg_mean_squared_error", cv=cv)

    ## predict on held out test
    y_pred = 100*pipeline.predict(X_test) #rescale age

    if y_test.ndim == 1: #single timepoint
        y_pred_BL = y_pred
        y_test_BL = y_test

        if y_test_FU is None:
            y_pred_FU = None 
        else:
            y_pred_FU = 100*pipeline.predict(X_test_FU) #rescale age

    else: # two timepoints
        y_pred_BL = y_pred[:,0]
        y_test_BL = y_test[:,0]
        y_pred_FU = y_pred[:,1]
        y_test_FU = y_test[:,1]
    
    perf_BL_df = calculate_perf_metrics(y_pred_BL, y_test_BL, visit="BL")

    if y_test_FU is None:
        perf_df = perf_BL_df
    else:
        perf_FU_df = calculate_perf_metrics(y_pred_FU, y_test_FU, visit="FU")
        
        perf_df = pd.concat([perf_BL_df,perf_FU_df], axis=1)

        perf_df["delta_test"] = y_test_FU - y_test_BL
        perf_df["delta_pred"] = y_pred_FU - y_pred_BL
    
    return perf_df, pipeline
