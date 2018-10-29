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

def learning_curves(X, Y, alg1, alg2, alg3, CV, score, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, decision tree, and support vector methods), varying
    the training set size for each prediction category: reactor type, cooling
    time, enrichment, and burnup

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
    *learning_curve.csv : csv file with learning curve results for each 
                          prediction category

    """    
    
    trainset_frac = np.array( [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 
                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] )
    col_names = ['AbsTrainSize', 'TrainScore', 'CV-Score']
    tsze = 'AbsTrainSize'
    tscr = 'TrainScore'
    cscr = 'CV-Score'
    tstd = 'TrainStd'
    cstd = 'CV-Std'

    # knn
    tsize, train, cv = learning_curve(alg1, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=4)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df1 = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                        cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    df1['Algorithm'] = 'knn'

    # dtree
    tsize, train, cv = learning_curve(alg2, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=4)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df2 = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                        cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    df2['Algorithm'] = 'dtree'
    
    # svr
    tsize, train, cv = learning_curve(alg3, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=4)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df3 = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                        cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    df3['Algorithm'] = 'svr'

    lc_data = pd.concat([df1, df2, df3])
    lc_data.index.name = 'TrainSizeFrac'
    lc_data.to_csv(csv_name + '_learning_curve.csv')
    return 

def main():
    """
    Given training data, this script performs the generation of learning curves
    for three algorithms: knn, decision trees, support vectors.  Learning
    curves provide the prediction score/accuracy with respect to training set
    size

    """
    # which reactor parameter we are predicting
    label = sys.argv[1]

    trainX, trainY, csv_name = get_data(label)
    validation_inits = False
    knn_init, dtr_init, svr_init, kfold, score = init_learners(label, validation_inits)

    # learning curves for 3 algorithms
    learning_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
    
    return

if __name__ == "__main__":
    main()
