from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale
from sklearn import metrics
import numpy as np
import pandas as pd

def classification(X, Y, k, cv_fold):
    """
    Training for Classification: predicts a model using two knn algorithms, 
    returning the training and cross validation accuracies for each. 
    
    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
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

def regression(X, Y, k, cv_fold):
    """ 
    Training for Regression: predicts a model using two knn algorithms, 
    returning the training and cross validation errors for each. 

    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    l1 = KNeighborsRegressor(n_neighbors = k, metric='l1', p=1)
    l2 = KNeighborsRegressor(n_neighbors = k, metric='l2', p=2)
    l1_pred = cross_val_predict(l1, X, Y, cv=cv_fold)
    l2_pred = cross_val_predict(l2, X, Y, cv=cv_fold)
    # for diagnostic curves
    l1.fit(X, Y)
    l2.fit(X, Y)
    tr1_pred = l1.predict(X)
    tr2_pred = l2.predict(X)
    # Errors
    cvl1 = metrics.mean_squared_error(Y, l1_pred)
    cvl2 = metrics.mean_squared_error(Y, l2_pred)
    trl1 = metrics.mean_squared_error(Y, tr1_pred)
    trl2 = metrics.mean_squared_error(Y, tr2_pred)
    # mean squared error for csv
    mse = (cvl1, trl1, cvl2, trl2)
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

    Outputs
    -------
    reactor.csv : accuracy for l1nn & l2nn
    cooling.csv : error for l1nn & l2nn
    enrichment.csv : error for l1nn & l2nn
    burnup.csv : error for l1nn & l2nn

    """
    
    # regularization parameters (k's differ for each type of prediction)
    #reg = {'r_l1' : 15, 'r_l2' : 20, 
    #       'c_l1' : 7, 'c_l2' : 10, 
    #       'e_l1' : 7, 'e_l2' : 10, 
    #       'b_l1' : 30, 'b_l2' : 35,
    #       }
    cv_folds = 10
    klist = list(range(1, 50))
    reactor_acc = []
    cooling_err = []
    enrichment_err = []
    burnup_err = []
    for k in klist:
        r = classification(X, rY, k, cv_folds)
        c = regression(X, cY, k, cv_folds)
        e = regression(X, eY, k, cv_folds)
        b = regression(X, bY, k, cv_folds)
        reactor_acc.append(r)
        cooling_err.append(c)
        enrichment_err.append(e)
        burnup_err.append(b)


#            r_sum = [sum(x) for x in zip(r, r_sum)]
#            c_sum = [sum(x) for x in zip(c, c_sum)]
#            e_sum = [sum(x) for x in zip(e, e_sum)]
#            b_sum = [sum(x) for x in zip(b, b_sum)]
#        # Take Average
#        reactor = [x / n_trials for x in r_sum]
#        cooling = [x / n_trials for x in c_sum]
#        enrichment = [x / n_trials for x in e_sum]
#        burnup = [x / n_trials for x in b_sum]
#        reactor_acc.append(reactor)
#        cooling_err.append(cooling)
#        enrichment_err.append(enrichment)
#        burnup_err.append(burnup)
    
    # Save results
    cols = ['CVL1', 'TrainL1', 'CVL2', 'TrainL2']
    pd.DataFrame(reactor_acc, columns=cols, index=klist).to_csv('reactor.csv')
    pd.DataFrame(cooling_err, columns=cols, index=klist).to_csv('cooling.csv')
    pd.DataFrame(enrichment_err, columns=cols, index=klist).to_csv('enrichment.csv')
    pd.DataFrame(burnup_err, columns=cols, index=klist).to_csv('burnup.csv')

    return 
