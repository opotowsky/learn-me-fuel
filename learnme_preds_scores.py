#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold

import pandas as pd
import numpy as np

def errors_and_scores(trainX, Y, knn_init, rr_init, svr_init, rxtr_pred, scores, CV, subset, trainset):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    """
    # init/empty the lists
    knn_scores = []
    rr_scores = []
    svr_scores = []
    for alg in ('knn', 'rr', 'svr'):
        # get init'd learner
        if alg == 'knn':
            alg_pred = knn_init
        elif alg == 'rr':
            alg_pred = rr_init
        else:
            alg_pred = svr_init
        # cross valiation to obtain scores
        cv_info = cross_validate(alg_pred, trainX, Y, scoring=scores, cv=CV, return_train_score=False)
        df = pd.DataFrame(cv_info)
        if rxtr_pred is not 'reactor':
            # to get MSE -> RMSE
            test_mse = df['test_neg_mean_squared_error']
            rmse = lambda x : -1 * np.sqrt(-1*x)
            df['test_neg_rmse'] = rmse(test_mse)
        # for concatting since it's finnicky
        if alg == 'knn':
            df['algorithm'] = 'knn'
            knn_scores = df
        elif alg == 'rr':
            df['algorithm'] = 'rr'
            rr_scores = df
        else:
            df['algorithm'] = 'svr'
            svr_scores = df
    cv_results = [knn_scores, rr_scores, svr_scores]
    df = pd.concat(cv_results)
    df.to_csv('trainset_' + trainset + '_' + subset + '_' + rxtr_pred + '_scores.csv')
    return

def main():
    """
    Given training data, this script trains and tracks each prediction for
    several algorithms and saves the predictions and ground truth to a CSV file
    """
    # Parameters for the training and predictions
    CV = 10
    # The hand-picked numbers are based on the dayman test set validation curves
    k = 13
    a = 100
    g = 0.001
    c = 10000

    pkl_base = './pkl_trainsets/2jul2018/2jul2018_trainset'
    for trainset in ('1', '2'):
        for subset in ('fiss', 'act', 'fissact', 'all'):
            pkl = pkl_base + trainset + '_nucs_' + subset + '_not-scaled.pkl'
            trainXY = pd.read_pickle(pkl)
            trainX, rY, cY, eY, bY = splitXY(trainXY)
            if subset == 'all':
                top_n = 100
                nuc_set = top_nucs(trainX, top_n)
                trainX = filter_nucs(trainX, nuc_set, top_n)
            trainX = scale(trainX)
            
            # loops through each reactor parameter to do separate predictions
            for Y in ('c', 'e', 'b', 'r'):
                trainY = pd.Series()
                # get param names and set ground truth
                if Y == 'c':
                    trainY = cY
                    parameter = 'cooling'
                elif Y == 'e': 
                    trainY = eY
                    parameter = 'enrichment'
                elif Y == 'b':
                    trainY = bY
                    parameter = 'burnup'
                else:
                    trainY = rY
                    parameter = 'reactor'
                
                # initialize a learner
                if Y is not 'r':
                    scores = ['r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error']
                    kfold = KFold(n_splits=CV, shuffle=True)
                    knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    rr_init = Ridge(alpha=a)
                    svr_init = SVR(gamma=g, C=c)
                else:
                    scores = ['accuracy', ]
                    kfold = StratifiedKFold(n_splits=CV, shuffle=True)
                    knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
                    rr_init = RidgeClassifier(alpha=a, class_weight='balanced')
                    svr_init = SVC(gamma=g, C=c, class_weight='balanced')
                
                # make predictions
                knn = cross_val_predict(knn_init, trainX, y=trainY, cv=kfold)
                rr = cross_val_predict(rr_init, trainX, y=trainY, cv=kfold)
                svr = cross_val_predict(svr_init, trainX, y=trainY, cv=kfold)
                preds_by_alg = pd.DataFrame({'TrueY': trainY, 'kNN': knn, 
                                             'Ridge': rr, 'SVR': svr}, 
                                             index=trainY.index)
                preds_by_alg.to_csv('trainset_' + trainset + '_' + subset + '_' + parameter + '_predictions.csv')
                
                # calculate errors and scores
                errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, parameter, scores, kfold, subset, trainset)

    return

if __name__ == "__main__":
    main()
