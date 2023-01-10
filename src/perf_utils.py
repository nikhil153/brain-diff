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



def get_model_perf(model_dict,X_train, y_train, X_val, y_val):
    perf_df = pd.DataFrame()
    for model_name, model_instance in model_dict.items():        
        CV_scores, y_pred, test_loss1, test_loss2, test_r1, test_r2 = get_brain_age_perf(X_train, y_train, X_val, y_val, model_instance)
        
        df = pd.DataFrame()

        if y_val.ndim == 2:
            print("mutli-task model")
            df["age_at_ses2"] = y_val[:,0]
            df["age_at_ses3"] = y_val[:,1]
            df["brainage_at_ses2"] = y_pred[:,0]
            df["brainage_at_ses3"] = y_pred[:,1]
            df["ses2_sq_err"] = test_loss1
            df["ses3_sq_err"] = test_loss2
            df["ses2_abs_err"] = np.abs(df[f"brainage_at_ses2"] - df["age_at_ses2"])
            df["ses3_abs_err"] = np.abs(df[f"brainage_at_ses3"] - df["age_at_ses3"])
            df["ses2_r1"] = test_r1
            df["ses3_r1"] = test_r2
        
            df["model"] = model_name

            ses2_mae = df["ses2_abs_err"].mean()
            ses3_mae = df["ses3_abs_err"].mean()
            print(f"model: {model_name}, val mae: {ses2_mae:4.3f},{ses3_mae:4.3f}\t"
                    f"correlation: {test_r1:4.3f},{test_r2:4.3f}")

        else:
            df["age_at_ses2"] = y_val
            df[f"brainage_at_ses2"] = y_pred
            df["ses2_sq_err"] = test_loss1
            df["ses2_abs_err"] = np.abs(y_pred - y_val)
            df["ses2_r1"] = test_r1
            df["model"] = model_name

            ses2_mae = df["ses2_abs_err"].mean()
            print(f"model: {model_name}, val mae: {ses2_mae:4.3f}\t"
                    f"correlation: {test_r1:4.3f}")

        perf_df = perf_df.append(df)
        
    return perf_df

def get_brain_age_perf(X_CV, y_CV, X_test, y_test, model, cv=2):
    """ Compute CV score and heldout sample MAE and correlation. 
        This is used with baseline sklearn models.
    """
    y_CV = np.squeeze(y_CV)
    y_test = np.squeeze(y_test)

    y_CV = y_CV/100 #scale age

    pipeline = Pipeline([("brainage_model", model)])
    pipeline.fit(X_CV, y_CV)

    # Evaluate the models using crossvalidation
    CV_scores = cross_val_score(pipeline, X_CV, y_CV,
                                scoring="neg_mean_squared_error", cv=cv)

    ## predict on held out test
    y_pred = 100*pipeline.predict(X_test) #rescale age

    if y_test.ndim == 1: #single timepoint
        test_loss1 = (y_pred - y_test)**2
        test_r1 = stats.pearsonr(y_pred,y_test)[0]
        test_loss2 = None
        test_r2 = None
    else: # two timepoints
        test_loss1 = (y_pred[:,0] - y_test[:,0])**2
        test_r1 = stats.pearsonr(y_pred[:,0],y_test[:,0])[0]
        test_loss2 = (y_pred[:,1] - y_test[:,1])**2
        test_r2 = stats.pearsonr(y_pred[:,1],y_test[:,1])[0]
        
    return CV_scores, y_pred, test_loss1, test_loss2, test_r1, test_r2
