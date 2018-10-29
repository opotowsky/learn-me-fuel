#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs, get_algparams

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, validation_curve

import pandas as pd
import numpy as np
import sys

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
    depth_list = np.linspace(10, 100, 10).astype(int)
    feat_list = np.linspace(5, 47, 10).astype(int)
    gamma_list = np.logspace(-4, -1, 10)
    c_list = np.logspace(0, 5, 10)

    # knn
    train, cv = validation_curve(alg1, X, Y, 'n_neighbors', k_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df1 = pd.DataFrame({'ParamList' : k_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df1['Algorithm'] = 'knn'

    # dtree
    train, cv = validation_curve(alg2, X, Y, 'max_depth', depth_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df2 = pd.DataFrame({'ParamList' : depth_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df2['Algorithm'] = 'dtree'
    
    train, cv = validation_curve(alg2, X, Y, 'max_features', feat_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df3 = pd.DataFrame({'ParamList' : feat_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df3['Algorithm'] = 'dtree'
    
    # svr
    train, cv = validation_curve(alg3, X, Y, 'gamma', gamma_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df4 = pd.DataFrame({'ParamList' : gamma_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df4['Algorithm'] = 'svr'

    train, cv = validation_curve(alg3, X, Y, 'C', c_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df5 = pd.DataFrame({'ParamList' : c_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df5['Algorithm'] = 'svr'

    vc_data = pd.concat([df1, df2, df3, df4, df5])
    vc_data.to_csv(csv_name + '_validation_curve.csv')
    return 


def main():
    """
    Given training data, this script performs the generation of learning curves
    for three algorithms: knn, decision trees, support vectors.  Learning
    curves provide the prediction score/accuracy with respect to training set
    size

    """
    #pkl = '../small_trainset.pkl'
    pkl = './small_trainset.pkl'
    
    # Parameters for the training and predictions
    CV = 10

    # Get optimized algorithm parameters for specific prediction case
    label = sys.argv[1]
    k, depth, feats, g, c = get_algparams(label)
    # trainX and trainY
    trainXY = pd.read_pickle(pkl)
    # hyperparam optimization was done on 60% of training set
    #trainXY = trainXY.sample(frac=0.6)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    if label == 'cooling':
        trainY = cY
    elif label == 'enrichment': 
        trainY = eY
    elif label == 'reactor':
        trainY = rY
    else: # label == 'burnup'
        # burnup needs much less training data...this is 24% of data set
        #trainXY = trainXY.sample(frac=0.4)
        #trainX, rY, cY, eY, bY = splitXY(trainXY)
        #trainX = scale(trainX)
        trainY = bY
    csv_name = 'fissact_m60_' + label

    # initialize learners
    score = 'explained_variance'
    kfold = KFold(n_splits=CV, shuffle=True)
    knn_init = KNeighborsRegressor(weights='distance')
    dtr_init = DecisionTreeRegressor()
    svr_init = SVR()
    if Y is 'r':
        score = 'accuracy'
        kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsClassifier(weights='distance')
        dtr_init = DecisionTreeClassifier(class_weight='balanced')
        svr_init = SVC(class_weight='balanced')
    
    # validation curves for 3 algorithms
    validation_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)

    return

if __name__ == "__main__":
    main()
