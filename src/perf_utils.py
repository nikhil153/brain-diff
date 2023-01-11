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


def get_brain_age_perf(model, X_CV, y_CV, X_test, y_test, X_test_FU=None, y_test_FU=None, cv=2):
    """ Compute CV score and heldout sample MAE and correlation. 
        This is used with baseline sklearn models.
    """
    y_CV = np.squeeze(y_CV)
    y_test = np.squeeze(y_test)

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
        
    # Calculate various perf metrics
    sq_err_BL = (y_pred_BL - y_test_BL)**2
    abs_err_BL = np.abs(y_pred_BL - y_test_BL)
    corr_BL = stats.pearsonr(y_pred_BL, y_test_BL)[0]

    if y_test_FU is None:
        sq_err_FU = None
        abs_err_FU = None
        corr_FU = None
        delta_test = None
        delta_pred = None
    else:
        sq_err_FU = (y_pred_FU - y_test_FU)**2
        abs_err_FU = abs(y_pred_FU - y_test_FU)
        corr_FU = stats.pearsonr(y_pred_FU, y_test_FU)[0]
        delta_test = y_test_FU - y_test_BL
        delta_pred = y_pred_FU - y_pred_BL
    

    df = pd.DataFrame()
    df["y_test_BL"] = y_test_BL # First visit perf
    df["y_pred_BL"] = y_pred_BL
    df["sq_err_BL"] = sq_err_BL
    df["abs_err_BL"] = abs_err_BL
    df["corr_BL"] = corr_BL

    df["y_test_FU"] = y_test_FU # Second visit perf
    df["y_pred_FU"] = y_pred_FU
    df["sq_err_FU"] = sq_err_FU
    df["abs_err_FU"] = abs_err_FU
    df["corr_FU"] = corr_FU
    df["delta_test"] = delta_test
    df["delta_pred"] = delta_pred
    # df["CV_scores"] = tuple(CV_scores)

    return df
