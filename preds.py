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

def classification(X, y, k, cv_fold):
    """
    Training for Classification: predicts a model using two knn algorithms, 
    returning the training and cross validation accuracies for each. 
    
    """
    
    # m is training set size
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    l1 = KNeighborsClassifier(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsClassifier(n_neighbors = k, metric='l2', p=2)
    _, trl1, cvl1 = learning_curve(l1, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold)
    m, trl2, cvl2 = learning_curve(l2, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold)
    accuracy = format(m, cvl1, trl1, cvl2, trl2)
    return accuracy

def regression(X, y, k, cv_fold):
    """ 
    Training for Regression: predicts a model using two knn algorithms, 
    returning the training and cross validation errors for each. 

    """
    # m is training set size
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    l1 = KNeighborsRegressor(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsRegressor(n_neighbors = k, metric='l2', p=2)
    _, trl1, cvl1 = learning_curve(l1, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold, scoring='neg_mean_squared_error')
    m, trl2, cvl2 = learning_curve(l2, X, y, train_sizes=TRAINSET_SIZES, cv=cv_fold, scoring='neg_mean_squared_error')
    mse = format(m, cvl1, trl1, cvl2, trl2)
    return mse

def train_and_predict(X, rY, cY, eY, bY):
    """
    Given training and testing data, this script runs some ML algorithms
    (currently, this is k nearest neighbors with 2 distance metrics) 
    for each prediction category: reactor type, cooling time, enrichment, 
    and burnup

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    *Y : series with labels for training data

    Returns
    -------
    reactor.csv : accuracy for l1nn & l2nn
    cooling.csv : error for l1nn & l2nn
    enrichment.csv : error for l1nn & l2nn
    burnup.csv : error for l1nn & l2nn

    """

    cv_folds = 10
    k = 2
    trainsize = TRAINSET_SIZES
    reactor_acc  = classification(X, rY, k, cv_folds)
    cooling_err= regression(X, cY, k, cv_folds)
    enrichment_err = regression(X, eY, k, cv_folds)
    burnup_err = regression(X, bY, k, cv_folds)
    # Save results
    cols = ['CVL1', 'TrainL1', 'CVL2', 'TrainL2']
    pd.DataFrame(reactor_acc, columns=cols, index=trainsize).to_csv('reactor.csv')
    pd.DataFrame(cooling_err, columns=cols, index=trainsize).to_csv('cooling.csv')
    pd.DataFrame(enrichment_err, columns=cols, index=trainsize).to_csv('enrichment.csv')
    pd.DataFrame(burnup_err, columns=cols, index=trainsize).to_csv('burnup.csv')

    return 
