#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs
from scipy.stats import expon, uniform

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold, RandomizedSearchCV, learning_curve, validation_curve

import pandas as pd
import numpy as np

def validation_curves(X, Y, alg1, alg2, alg3, CV, score, csv_name):
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
    score : 
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *validation_curve.csv : csv file with val curve results for each 
                            prediction category

    """    
    
    # Note: I'm trying to avoid loops here so the code is inelegant

    # Varied alg params for validation curves
    k_list = np.linspace(1, 39, 10).astype(int)
    depth_list = np.linspace(10, 100, 10)
    gamma_list = np.logspace(-4, -1, 10)
    c_list = np.linspace(0.1, 100000, 10)

    # knn
    train, cv = validation_curve(alg1, X, Y, 'n_neighbors', k_list, cv=CV, 
                                 scoring=score)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df1 = pd.DataFrame({'ParamList' : k_list, 'TrainScore' : train_mean, 'CV-Score' : cv_mean})
    df1['Algorithm'] = 'knn'

    # dtree
    train, cv = validation_curve(alg2, X, Y, 'max_depth', depth_list, cv=CV, 
                                 scoring=score)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df2 = pd.DataFrame({'ParamList' : depth_list, 'TrainScore' : train_mean, 'CV-Score' : cv_mean})
    df2['Algorithm'] = 'dtree'
    
    # svr
    train, cv = validation_curve(alg3, X, Y, 'gamma', gamma_list, cv=CV, 
                                 scoring=score)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df3 = pd.DataFrame({'ParamList' : gamma_list, 'TrainScore' : train_mean, 'CV-Score' : cv_mean})
    df3['Algorithm'] = 'svr'

    vc_data = pd.concat([df1, df2, df3])
    vc_data.to_csv(csv_name + '_validation_curve.csv')
    return 

def learning_curves(X, Y, alg1, alg2, alg3, CV, csv_name):
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
    
    # Note: I'm trying to avoid loops here so the code is inelegant

    trainset_frac = np.array( [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 
                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] )
    col_names = ['AbsTrainSize', 'TrainScore', 'CV-Score']
    tsze = 'AbsTrainSize'
    tscr = 'TrainScore'
    cscr = 'CV-Score'

    # knn
    tsize, train, cv = learning_curve(alg1, X, Y, train_sizes=trainset_frac, 
                                      cv=CV, shuffle=True)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df1 = pd.DataFrame({tsze : tsize, tscr : train_mean, cscr : cv_mean}, 
                        index=trainset_frac)
    df1['Algorithm'] = 'knn'

    # dtree
    tsize, train, cv = learning_curve(alg2, X, Y, train_sizes=trainset_frac, 
                                      cv=CV, shuffle=True)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df2 = pd.DataFrame({tsze : tsize, tscr : train_mean, cscr : cv_mean}, 
                        index=trainset_frac)
    df2['Algorithm'] = 'dtree'
    
    # svr
    tsize, train, cv = learning_curve(alg3, X, Y, train_sizes=trainset_frac, 
                                      cv=CV, shuffle=True)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df3 = pd.DataFrame({tsze : tsize, tscr : train_mean, cscr : cv_mean}, 
                        index=trainset_frac)
    df3['Algorithm'] = 'svr'

    lc_data = pd.concat([df1, df2, df3])
    lc_data.index.name = 'TrainSizeFrac'
    lc_data.to_csv(csv_name + '_learning_curve.csv')
    return 


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
    knn = cross_val_predict(alg1, X, y=Y, cv=CV)
    dtr = cross_val_predict(alg2, X, y=Y, cv=CV)
    svr = cross_val_predict(alg3, X, y=Y, cv=CV)

    preds_by_alg = pd.DataFrame({'TrueY': Y, 'kNN': knn, 
                                 'DTree': dtr, 'SVR': svr}, 
                                 index=trainY.index)
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
    
    cv_scr = cross_validate(alg1, X, Y, scoring=scores, cv=CV, return_train_score=False)
    df1 = pd.DataFrame(cv_scr)
    df1['Algorithm'] = 'knn'
    
    cv_scr = cross_validate(alg2, X, Y, scoring=scores, cv=CV, return_train_score=False)
    df2 = pd.DataFrame(cv_scr)
    df2['Algorithm'] = 'dtree'
    
    cv_scr = cross_validate(alg3, X, Y, scoring=scores, cv=CV, return_train_score=False)
    df3 = pd.DataFrame(cv_scr)
    df3['Algorithm'] = 'svr'
    
    cv_results = [df1, df2, df3]
    df = pd.concat(cv_results)
    df.to_csv(csv_name + '_scores.csv')
    
    return

def main():
    """
    Given training data, this script performs a number of ML tasks for three
    algorithms:
    1. errors_and_scores provides the prediction accuracy for each algorithm
    for each CV fold
    2. train_and_predict provides the prediction of each training instance
    3. validation_curves provides the prediction accuracy with respect to
    algorithm hyperparameter variations
    4. learning_curves  provides the prediction accuracy with respect to
    training set size

    """
    pkl = './pkl_trainsets/2jul2018/22jul2018_trainset3_nucs_fissact_not-scaled.pkl'
    
    # Parameters for the training and predictions
    CV = 5
    
    trainXY = pd.read_pickle(pkl)
    trainXY = trainXY.sample(frac=0.5)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    
    # loops through each reactor parameter to do separate predictions
    for Y in ('r', 'b', 'c', 'e'):
        trainY = pd.Series()
        # get param names and set ground truth
        if Y == 'c':
            trainY = cY
            parameter = 'cooling'
            k = 4
            depth = 25
            feats = 25 
            g = 0.001
            c = 200
        elif Y == 'e': 
            trainY = eY
            parameter = 'enrichment'
            k = 4
            depth = 25
            feats = 25 
            g = 0.001
            c = 200
        elif Y == 'b':
            trainY = bY
            parameter = 'burnup'
            k = 4
            depth = 25
            feats = 25 
            g = 0.001
            c = 200
        else:
            trainY = rY
            parameter = 'reactor'
            k = 4
            depth = 25
            feats = 25 
            g = 0.001
            c = 200
        
        csv_name = 'trainset3_fissact_m50_' + parameter
        
        # initialize learners
        score = 'explained_variance'
        kfold = KFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
        dtr_init = DecisionTreeRegressor(max_depth=depth, max_features=feats)
        svr_init = SVR(gamma=g, C=c)
        if Y is 'r':
            score = 'accuracy'
            kfold = StratifiedKFold(n_splits=CV, shuffle=True)
            knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
            dtr_init = DecisionTreeClassifier(max_depth=depth, max_features=feats, class_weight='balanced')
            svr_init = SVC(gamma=g, C=c, class_weight='balanced')

        # track predictions 
        track_predictions(trainX, trainY, knn_init, rr_init, svr_init, kfold, csv_name)

        # calculate errors and scores
        scores = ['explained_variance', 'neg_mean_absolute_error']
        if Y is 'r':
            scores = ['accuracy', ]
        errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, scores, kfold, csv_name)

        # learning curves
        learning_curves(trainX, trainY, knn_init, rr_init, svr_init, kfold, csv_name)
        
        # validation curves
        validation_curves(trainX, trainY, knn_init, rr_init, svr_init, kfold, score, csv_name)
        
        print("The {} predictions are complete\n".format(parameter), flush=True)

    return

if __name__ == "__main__":
    main()
