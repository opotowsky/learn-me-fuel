from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
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
    acc_l1 = metrics.accuracy_score(expected, predict1)
    acc_l2 = metrics.accuracy_score(expected, predict2)
    acc_rc = metrics.accuracy_score(expected, predict3)
    #acc_ann = ann_classification(trainX, trainY, testX, expected)
    #accuracy = (acc_l1, acc_l2, acc_rc, acc_ann)
    accuracy = (acc_l1, acc_l2, acc_rc)

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
    err_l1 = sqrt(metrics.mean_squared_error(expected, predict1))
    err_l2 = sqrt(metrics.mean_squared_error(expected, predict2))
    err_rr = sqrt(metrics.mean_squared_error(expected, predict3))
    #err_ann = ann_regression(trainX, trainY, testX, expected)
    #rmse = (err_l1, err_l2, err_rr, err_ann)
    rmse = (err_l1, err_l2, err_rr)

    return rmse

def train_and_predict(train, test):
    """
    Add deets, saves csv files

    Parameters
    ----------
    train :
    test : 

    Outputs
    -------
    reactor.csv : accuracy for 1nn, l2nn, ridge, ann
    enrichment.csv : RMSE for 1nn, l2nn, ridge, ann
    burnup.csv : RMSE for 1nn, l2nn, ridge, ann

    """
    
    # Add random errors of varying percents to nuclide vectors in the test set 
    # to mimic measurement error
    percent_err = np.arange(0.0, 10.25, 0.25)
    n_trials = 50
    reactor_acc = []
    enrichment_err = []
    burnup_err = []
    for err in percent_err:
        # warning: weak point
        r_sum = (0, 0, 0)
        e_sum = (0, 0, 0)
        b_sum = (0, 0, 0)
        for n in range(0, n_trials):
            # Add Error
            test.nuc_concs = random_error(err, test.nuc_concs)
            # Predict
            r = classification(train.nuc_concs, train.reactor, 
                               test.nuc_concs, test.reactor)
            e = regression(train.nuc_concs, train.enrichment, 
                           test.nuc_concs, test.enrichment)
            b = regression(train.nuc_concs, train.burnup, 
                           test.nuc_concs, test.burnup)
            r_sum = [sum(x) for x in zip(r, r_sum)]
            e_sum = [sum(x) for x in zip(e, e_sum)]
            b_sum = [sum(x) for x in zip(b, b_sum)]
        # Take Average
        reactor = [x / n_trials for x in r_sum]
        enrichment = [x / n_trials for x in e_sum]
        burnup = [x / n_trials for x in b_sum]
        reactor_acc.append(reactor)
        enrichment_err.append(enrichment)
        burnup_err.append(burnup)
    
    # Save results
    cols = ['L1NN', 'L2NN', 'RIDGE']
    #cols = ['L1NN', 'L2NN', 'RIDGE', 'ANN']
    pd.DataFrame(reactor_acc, columns=cols, index=percent_err).to_csv('reactor.csv')
    pd.DataFrame(enrichment_err, columns=cols, index=percent_err).to_csv('enrichment.csv')
    pd.DataFrame(burnup_err, columns=cols, index=percent_err).to_csv('burnup.csv')

    return 
