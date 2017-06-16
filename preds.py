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
    error = mean_absolute_percentage_error(expected, predict)
    
    return error

def mean_absolute_percentage_error(true, pred):
    """
    """

    mape = np.mean(np.abs((true - pred) / true)) * 100

    return mape

def classification(trainX, trainY, testX, testY):
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

    # Predictions for training, testing, and cross validation 
    train_predict1 = l1.predict(trainX)
    train_predict2 = l2.predict(trainX)
    train_predict3 = rc.predict(trainX)
    test_predict1 = l1.predict(testX)
    test_predict2 = l2.predict(testX)
    test_predict3 = rc.predict(testX)
    cv_predict1 = cross_val_predict(l1, trainX, trainY, cv = 5)
    cv_predict2 = cross_val_predict(l2, trainX, trainY, cv = 5)
    cv_predict3 = cross_val_predict(rc, trainX, trainY, cv = 5)
    train_acc1 = metrics.accuracy_score(trainY, train_predict1)
    train_acc2 = metrics.accuracy_score(trainY, train_predict2)
    train_acc3 = metrics.accuracy_score(trainY, train_predict3)
    test_acc1 = metrics.accuracy_score(testY, test_predict1)
    test_acc2 = metrics.accuracy_score(testY, test_predict2)
    test_acc3 = metrics.accuracy_score(testY, test_predict3)
    cv_acc1 = metrics.accuracy_score(trainY, cv_predict1)
    cv_acc2 = metrics.accuracy_score(trainY, cv_predict2)
    cv_acc3 = metrics.accuracy_score(trainY, cv_predict3)
    #acc_ann = ann_classification(trainX, trainY, testX, expected)
    accuracy = (train_acc1, train_acc2, train_acc3, test_acc1, test_acc2, 
                test_acc3, cv_acc1, cv_acc2, cv_acc3)

    return accuracy

def regression(trainX, trainY, testX, testY):
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
    
    # Predictions for training, testing, and cross validation 
    train_predict1 = l1.predict(trainX)
    train_predict2 = l2.predict(trainX)
    train_predict3 = rr.predict(trainX)
    test_predict1 = l1.predict(testX)
    test_predict2 = l2.predict(testX)
    test_predict3 = rr.predict(testX)
    cv_predict1 = cross_val_predict(l1, trainX, trainY, cv = 5)
    cv_predict2 = cross_val_predict(l2, trainX, trainY, cv = 5)
    cv_predict3 = cross_val_predict(rr, trainX, trainY, cv = 5)
    train_err1 = mean_absolute_percentage_error(trainY, train_predict1)
    train_err2 = mean_absolute_percentage_error(trainY, train_predict2)
    train_err3 = mean_absolute_percentage_error(trainY, train_predict3)
    test_err1 = mean_absolute_percentage_error(testY, test_predict1)
    test_err2 = mean_absolute_percentage_error(testY, test_predict2)
    test_err3 = mean_absolute_percentage_error(testY, test_predict3)
    cv_err1 = mean_absolute_percentage_error(trainY, cv_predict1)
    cv_err2 = mean_absolute_percentage_error(trainY, cv_predict2)
    cv_err3 = mean_absolute_percentage_error(trainY, cv_predict3)
    #err_ann = ann_regression(trainX, trainY, testX, expected)
    rmse = (train_err1, train_err2, train_err3, test_err1, test_err2, test_err3, 
            cv_err1, cv_err2, cv_err3)

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
    reactor : tuple of accuracy 
    enrichment : tuple of RMSE 
    burnup : tuple of RMSE 

    """
    
    reactor = classification(train.nuc_concs, train.reactor, 
                             test.nuc_concs, test.reactor)
    enrichment = regression(train.nuc_concs, train.enrichment, 
                            test.nuc_concs, test.enrichment)
    burnup = regression(train.nuc_concs, train.burnup, 
                        test.nuc_concs, test.burnup)
    
    return reactor, enrichment, burnup 
