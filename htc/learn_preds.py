#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs, get_algparams, get_data, init_learners

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, validation_curve

import pandas as pd
import numpy as np
import sys

def track_predictions(X, Y, alg1, alg2, alg3, CV, csv_name):
    """
    Saves csv's with predictions of each reactor parameter instance.
    
    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *predictions.csv : csv file with prediction results 

    """
    knn = cross_val_predict(alg1, X, y=Y, cv=CV, n_jobs=-1)
    dtr = cross_val_predict(alg2, X, y=Y, cv=CV, n_jobs=-1)
    svr = cross_val_predict(alg3, X, y=Y, cv=CV, n_jobs=-1)

    preds_by_alg = pd.DataFrame({'TrueY': Y, 'kNN': knn, 
                                 'DTree': dtr, 'SVR': svr}, 
                                 index=Y.index)
    preds_by_alg.to_csv(csv_name + '_predictions.csv')
    return

def errors_and_scores(X, Y, alg1, alg2, alg3, scores, CV, csv_name):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
    scores : list of scoring types (from sckikit-learn)
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *scores.csv : csv file with scores for each CV fold
    
    """
    
    cv_scr = cross_validate(alg1, X, Y, scoring=scores, cv=CV, 
                            return_train_score=False, n_jobs=-1)
    df1 = pd.DataFrame(cv_scr)
    df1['Algorithm'] = 'knn'
    
    cv_scr = cross_validate(alg2, X, Y, scoring=scores, cv=CV, 
                            return_train_score=False, n_jobs=-1)
    df2 = pd.DataFrame(cv_scr)
    df2['Algorithm'] = 'dtree'
    
    cv_scr = cross_validate(alg3, X, Y, scoring=scores, cv=CV, 
                            return_train_score=False, n_jobs=-1)
    df3 = pd.DataFrame(cv_scr)
    df3['Algorithm'] = 'svr'
    
    cv_results = [df1, df2, df3]
    df = pd.concat(cv_results)
    df.to_csv(csv_name + '_scores.csv')
    
    return

def main():
    """
    Given training data, this script performs the training and prediction for
    three algorithms: knn, decision trees, support vectors. one file tracks all
    the predictions and the other just tracks the scores for each cross
    validation fold

    """
    # which reactor parameter we are predicting
    label = sys.argv[1]

    trainX, trainY, csv_name = get_data(label)
    validation_inits = False
    knn_init, dtr_init, svr_init, kfold, score = init_learners(label, validation_inits)

    # predictions for 3 algorithms
    track_predictions(trainX, trainY, knn_init, dtr_init, svr_init, kfold, csv_name)

    # scores for 3 algorithms
    errors_and_scores(trainX, trainY, knn_init, dtr_init, svr_init, score, kfold, csv_name) 
    
    return

if __name__ == "__main__":
    main()
