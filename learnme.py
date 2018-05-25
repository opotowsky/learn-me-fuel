#! /usr/bin/env/ python

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn import metrics
import datetime as dt
import numpy as np
import pandas as pd


TRAINSET_SIZES = np.linspace(0.1, 1.0, 19)

def format(m, cv1, tr1, cv2, tr2):
    # average cv folds + format
    m = m.tolist()
    cv1 = np.array([np.mean(row) for row in cv1]).tolist()
    tr1 = np.array([np.mean(row) for row in tr1]).tolist()
    cv2 = np.array([np.mean(row) for row in cv2]).tolist()
    tr2 = np.array([np.mean(row) for row in tr2]).tolist()
    score = []
    for i in m:
        idx = m.index(i)
        scr = (cv1[idx], tr1[idx], cv2[idx], tr2[idx])
        score.append(scr)
    return score

def classification(X, Y, clist, cv_fold):
    """
    Training for Classification: predicts models using two regularization params, 
    returning the training and cross validation accuracies for each. 
    
    """
    
    accuracy = []
    for c in clist:
        l1 = LogisticRegression(C=c, penalty='l1')
        l2 = LogisticRegression(C=c, penalty='l2')
        strat_cv = StratifiedKFold(n_splits=cv_fold, shuffle=True)
        # for logR bug:
        Xs, Ys = shuffle(X, Y)
        l1_pred = cross_val_predict(l1, Xs, Ys, cv=strat_cv)
        l2_pred = cross_val_predict(l2, Xs, Ys, cv=strat_cv)
        # for diagnostic curves 
        l1.fit(Xs, Ys)
        l2.fit(Xs, Ys)
        tr1_pred = l1.predict(Xs)
        tr2_pred = l2.predict(Xs)
        # Accuracies
        cvl1 = metrics.accuracy_score(Ys, l1_pred)
        cvl2 = metrics.accuracy_score(Ys, l2_pred)
        trl1 = metrics.accuracy_score(Ys, tr1_pred)
        trl2 = metrics.accuracy_score(Ys, tr2_pred)
        # accuracies for csv
        acc = (cvl1, trl1, cvl2, trl2)
        accuracy.append(acc)
    return accuracy

def lc_classification(X, Y, c, cv_fold):
    """
    Learning curve for classification: predicts models using two regularization 
    parameters, returning the training and cross validation accuracies for each 
    with respect to a given training set size
    
    """
    l1 = LogisticRegression(C=c, penalty='l1')
    l2 = LogisticRegression(C=c, penalty='l2')
    strat_cv = StratifiedKFold(n_splits=cv_fold, shuffle=True)
    # for logR bug:
    Xs, Ys = shuffle(X, Y)
    _, trl1, cvl1 = learning_curve(l1, Xs, Ys, train_sizes=TRAINSET_SIZES, 
                                   cv=strat_cv
                                   )
    m, trl2, cvl2 = learning_curve(l2, X, Ys, train_sizes=TRAINSET_SIZES, 
                                   cv=strat_cv
                                   )
    accuracy = format(m, cvl1, trl1, cvl2, trl2)
    return accuracy

def regression(X, Y, alist, cv_fold):
    """ 
    Training for Regression: predicts a model using two regularization parameters 
    for optimization, returning the training and cross validation errors for each. 

    """
    mse = []
    for a in alist:
        l1 = ElasticNet(alpha=a, l1_ratio=1, selection='random')
        l2 = ElasticNet(alpha=a, l1_ratio=0, selection='random')
        l1_pred = cross_val_predict(l1, X, Y, cv=cv_fold)
        l2_pred = cross_val_predict(l2, X, Y, cv=cv_fold)
        # for diagnostic curves
        l1.fit(X, Y)
        l2.fit(X, Y)
        tr1_pred = l1.predict(X)
        tr2_pred = l2.predict(X)
        # Errors, negative for 'higher better' convention
        cvl1 = -1 * metrics.mean_squared_error(Y, l1_pred)
        cvl2 = -1 * metrics.mean_squared_error(Y, l2_pred)
        trl1 = -1 * metrics.mean_squared_error(Y, tr1_pred)
        trl2 = -1 * metrics.mean_squared_error(Y, tr2_pred)
        # mean squared error for csv
        mse_i = (cvl1, trl1, cvl2, trl2)
        mse.append(mse_i)
    return mse

def lc_regression(X, y, a, cv_fold):
    """ 
    Learning curve for regression: predicts models using two regularization 
    parameters, returning the training and cross validation accuracies for each 
    with respect to a given training set size

    """
    
    l1 = ElasticNet(alpha=a, l1_ratio=1, selection='random')
    l2 = ElasticNet(alpha=a, l1_ratio=0, selection='random')
    _, trl1, cvl1 = learning_curve(l1, X, y, train_sizes=TRAINSET_SIZES, 
                                   cv=cv_fold, scoring='neg_mean_squared_error'
                                   )
    m, trl2, cvl2 = learning_curve(l2, X, y, train_sizes=TRAINSET_SIZES, 
                                   cv=cv_fold, scoring='neg_mean_squared_error'
                                   )
    mse = format(m, cvl1, trl1, cvl2, trl2)
    return mse

def train_and_predict(X, rY, cY, eY, bY, train_src, nucs_tracked):
    """
    
    Given training and testing data, this script runs some ML algorithms
    (currently, this is 2 linear models with 2 error metrics) 
    for each prediction category: reactor type, cooling time, enrichment, 
    and burnup. Learning curves are also generated; comment out if unecessary

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    *Y : series with labels for training data

    Returns
    -------
    reactor*.csv : accuracy for l1 & l2 norms
    cooling*.csv : negative error for l1 & l2 norms
    enrichment*.csv : negative error for l1 & l2 norms
    burnup*.csv : negative error for l1 & l2 norms

    """    

    cols = ['CVL1', 'TrainL1', 'CVL2', 'TrainL2']
    cv_folds = 5

    #####################
    ## Learning Curves ##
    #####################
    alpha = 0.1
    rxtr_lc  = lc_classification(X, rY, alpha, cv_folds)
    cool_lc= lc_regression(X, cY, alpha, cv_folds)
    enr_lc = lc_regression(X, eY, alpha, cv_folds)
    burn_lc = lc_regression(X, bY, alpha, cv_folds)
    r_lc = 'lc_' + 'reactor' + nucs_tracked + train_src + '.csv'
    c_lc = 'lc_' + 'cooling' + nucs_tracked + train_src + '.csv'
    e_lc = 'lc_' + 'enrichment' + nucs_tracked + train_src + '.csv'
    b_lc = 'lc_' + 'burnup' + nucs_tracked + train_src + '.csv'
    idx_lc = TRAINSET_SIZES
    pd.DataFrame(rxtr_lc, columns=cols, index=idx_lc).to_csv(r_lc)
    pd.DataFrame(cool_lc, columns=cols, index=idx_lc).to_csv(c_lc)
    pd.DataFrame(enr_lc, columns=cols, index=idx_lc).to_csv(e_lc)
    pd.DataFrame(burn_lc, columns=cols, index=idx_lc).to_csv(b_lc)
    
    #######################
    ## Validation Curves ##
    #######################
    alist = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 
             0.8, 1, 1.3, 1.7, 2, 5, 10, 50, 100
             )
    rxtr_vc = classification(X, rY, alist, cv_folds)
    cool_vc = regression(X, cY, alist, cv_folds)
    enr_vc = regression(X, eY, alist, cv_folds)
    burn_vc = regression(X, bY, alist, cv_folds)

    idx_vc = alist
    r_vc = 'vc_' + 'reactor' + nucs_tracked + train_src + '.csv'
    c_vc = 'vc_' + 'cooling' + nucs_tracked + train_src + '.csv'
    e_vc = 'vc_' + 'enrichment' + nucs_tracked + train_src + '.csv'
    b_vc = 'vc_' + 'burnup' + nucs_tracked + train_src + '.csv'
    pd.DataFrame(rxtr_vc, columns=cols, index=idx_vc).to_csv(r_vc)
    pd.DataFrame(cool_vc, columns=cols, index=idx_vc).to_csv(c_vc)
    pd.DataFrame(enr_vc, columns=cols, index=idx_vc).to_csv(e_vc)
    pd.DataFrame(burn_vc, columns=cols, index=idx_vc).to_csv(b_vc)
    
    return 


def splitXY(dfXY, info):
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
    oY : dataframe with ORIGEN reactor name for each instance

    """

    if info == '_gammas':
        lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    else:
        lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor', 'total']
    dfX = dfXY.drop(lbls, axis=1)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    o_dfY = dfXY.loc[:, lbls[4]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY, o_dfY


def main():
    """
    Takes the pickle file of the training set, splits the dataframe into the 
    appropriate X and Ys for prediction of reactor type, cooling time, fuel 
    enrichment, and burnup. Scales the training data set for the algorithms, then 
    calls for the training and prediction. Saves all results as .csv files.
   
    """
    start = dt.datetime.now()
    print("Start Time: {}\n".format(start), flush=True)
    
    # Not dealing with gammas for now
    #info_src = ['_nucs', '_gammas']
    info_src = ['_nucs',]
    #subset = ['_fiss', '_act', '_fissact']
    # sub the subset for now
    subset = ['_fiss']
    for nucs_tracked in subset:
        print("Nuclide subset currently being used for training: {}\n".format(nucs_tracked), flush=True)
        for src in info_src:
            pkl_name = 'not-scaled_trainset' + src + nucs_tracked + '_8dec.pkl'
            trainXY = pd.read_pickle(pkl_name, compression=None)
            trainX, rY, cY, eY, bY = splitXY(trainXY, src)
            if info_src.index(src) == 0:
                trainX = scale(trainX)
            else:
                trainX = scale(trainX, with_mean=False)
            train_and_predict(trainX, rY, cY, eY, bY, src, nucs_tracked)
            print("Predictions for {} are complete\n".format(src), flush=True)
            print("{} Train & Predict Timestamp: {}\n".format(src, dt.datetime.now()), flush=True)
    print("Remember to move results to a dated directory!\n")
    total_time = dt.datetime.now() - start
    print("Time to run script: {}\n".format(total_time), flush=True)

if __name__ == "__main__":
    main()
