#! /usr/bin/env python3

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold
import pandas as pd
import numpy as np

def errors_and_scores(trainX, Y, knn_init, rr_init, svr_init, rxtr_pred, scores, CV, subset):
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
    df.to_csv('sfcompo_' + subset + rxtr_pred + '_scores.csv')
    return

def splitXY(dfXY):
    """
    Takes a dataframe with all X (features) and Y (labels) information and 
    produces five different pandas datatypes: a dataframe with nuclide info 
    only + a series for each label column.

    Parameters
    ----------
    dfXY : dataframe with nuclide concentraations and 4 labels: reactor type, 
           cooling time, enrichment, and burnup

    Returns
    -------
    dfX : dataframe with only nuclide concentrations for each instance
    rY : dataframe with reactor type for each instance
    cY : dataframe with cooling time for each instance
    eY : dataframe with fuel enrichment for each instance
    bY : dataframe with fuel burnup for each instance

    """

    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'total']
    dfX = dfXY.drop(lbls, axis=1)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY


def main():
    """
    Given training data, this script trains and tracks each prediction for
    several algorithms and saves the predictions and ground truth to a CSV file
    """

    pkl1 = './sfcompo_pickles/not-scaled_trainset_nucs_fiss_8dec.pkl'
    pkl2 = './sfcompo_pickles/not-scaled_trainset_nucs_act_8dec.pkl'
    pkl3 = './sfcompo_pickles/not-scaled_trainset_nucs_fissact_8dec.pkl'
    pkl_dict = {'fiss' : pkl1, 'act' : pkl2, 'fissact' : pkl3}
    for subset in ('fiss', 'act', 'fissact'):
        trainXY = pd.read_pickle(pkl_dict[subset])
        trainX, rY, cY, eY, bY = splitXY(trainXY)
        trainX = scale(trainX)
        
        CV = 10
        kfold = KFold(n_splits=CV, shuffle=True)
        scores = ['r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        # The hand-picked numbers are based on the dayman test set validation curves
        k = 13
        a = 100
        g = 0.001
        c = 10000
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
                scores = ['accuracy', ]
                kfold = StratifiedKFold(n_splits=CV, shuffle=True)
                trainY = rY
                parameter = 'reactor'
                ####precision = make_scorer(average_precision_score, average='weighted')
                ####scores = {'accuracy_score' : 'accuracy', 
                ####          'avg_precision' : precision
                ####          }
                #### can't do avg precision without some super crazy work b/c it's multiclass. 
                #### just balancing classes in the alg init below instead....
                #### should check if this makes accuracy an OK score in this case
            # initialize a learner
            if Y is not 'r':
                knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
                rr_init = Ridge(alpha=a)
                svr_init = SVR(gamma=g, C=c)
            else:
                knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
                rr_init = RidgeClassifier(alpha=a, class_weight='balanced')
                svr_init = SVC(gamma=g, C=c, class_weight='balanced')
            # make predictions
            #knn = cross_val_predict(knn_init, trainX, y=trainY, cv=kfold)
            #rr = cross_val_predict(rr_init, trainX, y=trainY, cv=kfold)
            #svr = cross_val_predict(svr_init, trainX, y=trainY, cv=kfold)
            #preds_by_alg = pd.DataFrame({'TrueY': trainY, 'kNN': knn, 
            #                             'Ridge': rr, 'SVR': svr}, 
            #                             index=trainY.index)
            #preds_by_alg.to_csv('sfcompo_' + subset + parameter + '_predictions.csv')
            # calculate errors and scores
            errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, parameter, scores, kfold, subset)
    return

if __name__ == "__main__":
    main()
