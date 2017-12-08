from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn import metrics
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

def classification(X, Y, k, cv_fold):
    """
    Training for Classification: predicts a model using two knn algorithms, 
    returning the training and cross validation accuracies for each. 
    
    """
    
    l1 = KNeighborsClassifier(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsClassifier(n_neighbors = k, metric='l2', p=2)
    l1_pred = cross_val_predict(l1, X, Y, cv=cv_fold)
    l2_pred = cross_val_predict(l2, X, Y, cv=cv_fold)
    # for diagnostic curves 
    l1.fit(X, Y)
    l2.fit(X, Y)
    tr1_pred = l1.predict(X)
    tr2_pred = l2.predict(X)
    # Accuracies
    cvl1 = metrics.accuracy_score(Y, l1_pred)
    cvl2 = metrics.accuracy_score(Y, l2_pred)
    trl1 = metrics.accuracy_score(Y, tr1_pred)
    trl2 = metrics.accuracy_score(Y, tr2_pred)
    # accuracies for csv
    accuracy = (cvl1, trl1, cvl2, trl2)
    return accuracy

def lc_classification(X, y, k, cv_fold):
    """
    Learning curve for classification: predicts a model using two knn algorithms, 
    returning the training and cross validation accuracies for each with respect
    to a given training set size
    
    """
    
    l1 = KNeighborsClassifier(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsClassifier(n_neighbors = k, metric='l2', p=2)
    _, trl1, cvl1 = learning_curve(l1, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold)
    m, trl2, cvl2 = learning_curve(l2, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold)
    accuracy = format(m, cvl1, trl1, cvl2, trl2)
    return accuracy

def regression(X, Y, k, cv_fold):
    """ 
    Training for Regression: predicts a model using two knn algorithms, 
    returning the training and cross validation errors for each. 

    """
    
    l1 = KNeighborsRegressor(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsRegressor(n_neighbors = k, metric='l2', p=2)
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
    mse = (cvl1, trl1, cvl2, trl2)
    return mse

def lc_regression(X, y, k, cv_fold):
    """ 
    Learning curve for regression: predicts a model using two knn algorithms,
    returning the training and cross validation errors for each with respect 
    to a given training set size

    """
    
    l1 = KNeighborsRegressor(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsRegressor(n_neighbors = k, metric='l2', p=2)
    _, trl1, cvl1 = learning_curve(l1, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold, scoring='neg_mean_squared_error')
    m, trl2, cvl2 = learning_curve(l2, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold, scoring='neg_mean_squared_error')
    mse = format(m, cvl1, trl1, cvl2, trl2)
    return mse

def train_and_predict(X, rY, cY, eY, bY, train_src):
    """
    Given training and testing data, this script runs some ML algorithms
    (currently, this is k nearest neighbors with 2 distance metrics) 
    for each prediction category: reactor type, cooling time, enrichment, 
    and burnup. Learning curves are also generated; comment out if unecessary

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    *Y : series with labels for training data

    Returns
    -------
    reactor*.csv : accuracy for l1nn & l2nn
    cooling*.csv : error for l1nn & l2nn
    enrichment*.csv : error for l1nn & l2nn
    burnup*.csv : error for l1nn & l2nn

    """

    #####################
    ## Learning Curves ##
    #####################
    cv_folds = 5
    k = 3
    reactor_lc  = lc_classification(X, rY, k, cv_folds)
    cooling_lc= lc_regression(X, cY, k, cv_folds)
    enrichment_lc = lc_regression(X, eY, k, cv_folds)
    burnup_lc = lc_regression(X, bY, k, cv_folds)
    
    #######################
    ## Validation Curves ##
    #######################
    cv_folds = 5
    klist = list(range(1, 100))
    reactor_vc = []
    cooling_vc = []
    enrichment_vc = []
    burnup_vc = []
    for k in klist:
        r = classification(X, rY, k, cv_folds)
        c = regression(X, cY, k, cv_folds)
        e = regression(X, eY, k, cv_folds)
        b = regression(X, bY, k, cv_folds)
        reactor_vc.append(r)
        cooling_vc.append(c)
        enrichment_vc.append(e)
        burnup_vc.append(b)

    ##############
    ## Save CSVs##
    ##############
    cols = ['CVL1', 'TrainL1', 'CVL2', 'TrainL2']
    idx = klist.append(TRAINSET_SIZES)
    rxtr = reactor_vc.append(reactor_lc)
    cool = cooling_vc.append(cooling_lc)
    enr = enrichment_vc.append(enrichment_lc)
    burn = burnup_vc.append(burnup_lc)
    rcsv = 'reactor' + train_src + '.csv'
    ccsv = 'cooling' + train_src + '.csv'
    ecsv = 'enrichment' + train_src + '.csv'
    bcsv = 'burnup' + train_src + '.csv'
    pd.DataFrame(rxtr, columns=cols, index=idx).to_csv(rcsv)
    pd.DataFrame(cool, columns=cols, index=idx).to_csv(ccsv)
    pd.DataFrame(enr, columns=cols, index=idx).to_csv(ecsv)
    pd.DataFrame(burn, columns=cols, index=idx).to_csv(bcsv)
    
    return 
