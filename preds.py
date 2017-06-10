from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  
from sklearn import metrics
from math import sqrt
import numpy as np
import pandas as pd

def random_error(percent_err, df):
    """
    Given a dataframe of nuclide vectors, add error to each element in each 
    nuclide vector that has a random value within the range [1-err, 1+err]

    Parameters
    ----------
    percent_err : a float indicating the maximum error that can be added to the nuclide 
                  vectors
    df : dataframe of only nuclide concentrations

    Returns
    -------
    df_err : dataframe with nuclide concentrations altered by some error

    """
    x = len(df)
    y = len(df.columns)
    err = percent_err / 100.0
    low = 1 - err
    high = 1 + err
    errs = np.random.uniform(low, high, (x, y))
    df_err = df * errs
    
    return df_err

def ann_classification(trainX, trainY, testX, expected):
    """
    
    """
    
    ann = MLPClassifier(hidden_layer_sizes=(500,), tol=0.01)
    ann.fit(trainX, trainY)
    predict = ann.predict(testX)
    accuracy = metrics.accuracy_score(expected, predict)
    
    return accuracy

def ann_regression(trainX, trainY, testX, expected):
    """
    
    """
    # Scale the data
    scaler = StandardScaler()  
    scaler.fit(trainX)  
    trainX = scaler.transform(trainX)  
    testX = scaler.transform(testX)

    ann = MLPRegressor(hidden_layer_sizes=(500,), tol=0.01)
    ann.fit(trainX, trainY)
    predict = ann.predict(testX)
    error = sqrt(metrics.mean_squared_error(expected, predict))
    
    return error

def classification(trainX, trainY, testX, expected):
    """
    Training for Classification
    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsClassifier(metric='l1', p=1)
    l2 = KNeighborsClassifier(metric='l2', p=2)
    rc = RidgeClassifier()
    l1.fit(trainX, trainY)
    l2.fit(trainX, trainY)
    rc.fit(trainX, trainY)
    
    # Predictions
    predict1 = l1.predict(testX)
    predict2 = l2.predict(testX)
    predict3 = rc.predict(testX)
    cv_predict1 = cross_val_predict(l1, trainX, trainY, cv = 5)
    cv_predict2 = cross_val_predict(l2, trainX, trainY, cv = 5)
    cv_predict3 = cross_val_predict(rc, trainX, trainY, cv = 5)
    acc_l1 = metrics.accuracy_score(expected, predict1)
    acc_l2 = metrics.accuracy_score(expected, predict2)
    acc_rc = metrics.accuracy_score(expected, predict3)
    cv_acc_l1 = metrics.accuracy_score(trainY, cv_predict1)
    cv_acc_l2 = metrics.accuracy_score(trainY, cv_predict2)
    cv_acc_rc = metrics.accuracy_score(trainY, cv_predict3)
    #acc_ann = ann_classification(trainX, trainY, testX, expected)
    #accuracy = (acc_l1, acc_l2, acc_rc, acc_ann)
    accuracy = (acc_l1, acc_l2, acc_rc, cv_acc_l1, cv_acc_l2, cv_acc_rc)

    return accuracy

def regression(trainX, trainY, testX, expected):
    """
    Training for Regression
    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsRegressor(metric='l1', p=1)
    l2 = KNeighborsRegressor(metric='l2', p=2)
    rr = Ridge()
    l1.fit(trainX, trainY)
    l2.fit(trainX, trainY)
    rr.fit(trainX, trainY)
    
    # Predictions
    predict1 = l1.predict(testX)
    predict2 = l2.predict(testX)
    predict3 = rr.predict(testX)
    cv_predict1 = cross_val_predict(l1, trainX, trainY, cv = 5)
    cv_predict2 = cross_val_predict(l2, trainX, trainY, cv = 5)
    cv_predict3 = cross_val_predict(rr, trainX, trainY, cv = 5)
    err_l1 = sqrt(metrics.mean_squared_error(expected, predict1))
    err_l2 = sqrt(metrics.mean_squared_error(expected, predict2))
    err_rr = sqrt(metrics.mean_squared_error(expected, predict3))
    cv_err_l1 = sqrt(metrics.mean_squared_error(trainY, cv_predict1))
    cv_err_l2 = sqrt(metrics.mean_squared_error(trainY, cv_predict2))
    cv_err_rr = sqrt(metrics.mean_squared_error(trainY, cv_predict3))
    #err_ann = ann_regression(trainX, trainY, testX, expected)
    #rmse = (err_l1, err_l2, err_rr, err_ann)
    rmse = (err_l1, err_l2, err_rr, cv_err_l1, cv_err_l2, cv_err_rr)

    return rmse

def train_and_predict(train, test):
    """
    Add deets, returns 

    Parameters
    ----------
    train :
    test : 

    Returns
    -------
    reactor : tuple of accuracy for 1nn, l2nn, ridge, ann
    enrichment : tuple of RMSE for 1nn, l2nn, ridge, ann
    burnup : tuple of RMSE for 1nn, l2nn, ridge, ann

    """
    
    reactor = classification(train.nuc_concs, train.reactor, 
                             test.nuc_concs, test.reactor)
    enrichment = regression(train.nuc_concs, train.enrichment, 
                            test.nuc_concs, test.enrichment)
    burnup = regression(train.nuc_concs, train.burnup, 
                        test.nuc_concs, test.burnup)
    
    return reactor, enrichment, burnup 
