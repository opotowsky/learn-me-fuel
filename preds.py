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
    Old code, keeping for future reference    
    """
    
    ann = MLPClassifier(hidden_layer_sizes=(500,), tol=0.01)
    ann.fit(trainX, trainY)
    predict = ann.predict(testX)
    accuracy = metrics.accuracy_score(expected, predict)
    
    return accuracy

def ann_regression(trainX, trainY, testX, expected):
    """
    Old code, keeping for future reference    
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
    Given 2 vectors of values, true and pred, calculate the MAP error

    """

    mape = np.mean(np.abs((true - pred) / true)) * 100

    return mape

def classification(trainX, trainY, testX, testY, k_l1, k_l2, a_rc):
    """
    Training for Classification: predicts a model using three algorithms, returning
    the training, testing, and cross validation accuracies for each. 
    
    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsClassifier(n_neighbors = k_l1, metric='l1', p=1)
    l2 = KNeighborsClassifier(n_neighbors = k_l2, metric='l2', p=2)
    rc = RidgeClassifier(alpha = a_rc)
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
    accuracy = (train_acc1, train_acc2, train_acc3, test_acc1, test_acc2, 
                test_acc3, cv_acc1, cv_acc2, cv_acc3)

    return accuracy

def regression(trainX, trainY, testX, testY, k_l1, k_l2, a_rr):
    """ 
    Training for Regression: predicts a model using three algorithms, returning
    the training error, testing error, and cross validation error for each. 

    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsRegressor(n_neighbors = k_l1, metric='l1', p=1)
    l2 = KNeighborsRegressor(n_neighbors = k_l2, metric='l2', p=2)
    rr = Ridge(alpha = a_rr)
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
    rmse = (train_err1, train_err2, train_err3, test_err1, test_err2, test_err3, 
            cv_err1, cv_err2, cv_err3)

    return rmse

def train_and_predict(train, test):
    """
    Given training and testing data, this script runs some ML algorithms
    (currently, this is nearest neighbors with 2 distance metrics and ridge
    regression) for each prediction category: reactor type, enrichment, and
    burnup

    Parameters 
    ---------- 
    
    train : group of dataframes that include training data and the three
            labels 
    test : group of dataframes that has the training instances and the three
           labels

    Returns
    -------
    reactor : tuple of accuracy for training, testing, and cross validation
              accuracies for all three algorithms
    enrichment : tuple of MAPE for training, testing, and cross validation
                 errors for all three algorithms
    burnup : tuple of MAPE for training, testing, and cross validation errors
             for all three algorithms

    """

    # regularization parameters (k and alpha (a) differ for each)
    reg = {'r_l1_k' : 15, 'r_l2_k' : 20, 'r_rc_a' : 0.1, 
           'e_l1_k' : 7, 'e_l2_k' : 10, 'e_rr_a' : 100, 
           'b_l1_k' : 30, 'b_l2_k' : 35, 'b_rr_a' : 100}

    reactor = classification(train.nuc_concs, train.reactor, 
                             test.nuc_concs, test.reactor, 
                             reg['r_l1_k'], reg['r_l2_k'], reg['r_rc_a'])
    enrichment = regression(train.nuc_concs, train.enrichment, 
                            test.nuc_concs, test.enrichment, 
                            reg['e_l1_k'], reg['e_l2_k'], reg['e_rr_a'])
    burnup = regression(train.nuc_concs, train.burnup, 
                        test.nuc_concs, test.burnup, 
                        reg['b_l1_k'], reg['b_l2_k'], reg['b_rr_a'])
    
    return reactor, enrichment, burnup 
