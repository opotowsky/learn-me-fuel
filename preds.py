from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def ann_classification(trainX, trainY, testX, testY):
    """
    Old code, keeping for future reference    
    """
    
    ann = MLPClassifier(hidden_layer_sizes=(500,), tol=0.01)
    ann.fit(trainX, trainY)
    predict = ann.predict(testX)
    accuracy = metrics.accuracy_score(testY, predict)
    
    return accuracy

def ann_regression(trainX, trainY, testX, testY):
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
    error = sqrt(metrics.mean_squared_error(testY, predict))
    
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
    
    # Predictions
    predict1 = l1.predict(testX)
    predict2 = l2.predict(testX)
    predict3 = rc.predict(testX)
    acc_l1 = metrics.accuracy_score(testY, predict1)
    acc_l2 = metrics.accuracy_score(testY, predict2)
    acc_rc = metrics.accuracy_score(testY, predict3)
    accuracy = (acc_l1, acc_l2, acc_rc)
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
    
    # Predictions
    predict1 = l1.predict(testX)
    predict2 = l2.predict(testX)
    predict3 = rr.predict(testX)
    err_l1 = mean_absolute_percentage_error(testY, predict1)
    err_l2 = mean_absolute_percentage_error(testY, predict2)
    err_rr = mean_absolute_percentage_error(testY, predict3)
    mape = (err_l1, err_l2, err_rr)

    return mape

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

    Outputs
    -------
    reactor.csv : accuracy for 1nn, l2nn, ridge
    enrichment.csv : MAPE for 1nn, l2nn, ridge
    burnup.csv : MAPE for 1nn, l2nn, ridge

    """
    
    # Apply PCA dimension reduction 
    #pca = PCA(n_components=50)
    #pca.fit(train.nuc_concs, train.burnup)
    #train.nuc_concs = pd.DataFrame(pca.transform(train.nuc_concs))
    # transform test set by the above pca fit as well
    #test.nuc_concs = pd.DataFrame(pca.transform(test.nuc_concs))
    #ncomp = pca.n_components_
    #print(ncomp)

    # regularization parameters (k and alpha (a) differ for each)
    reg = {'r_l1_k' : 1, 'r_l2_k' : 1, 'r_rc_a' : 0.1, 
           'e_l1_k' : 1, 'e_l2_k' : 1, 'e_rr_a' : 100, 
           'b_l1_k' : 1, 'b_l2_k' : 1, 'b_rr_a' : 100}
    
    # Add random errors of varying percents to nuclide vectors in the test set 
    # to mimic measurement error
    percent_err = np.arange(0.0, 10.25, 0.25)
    n_trials = 3 
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
                               test.nuc_concs, test.reactor, 
                               reg['r_l1_k'], reg['r_l2_k'], reg['r_rc_a'])
            e= regression(train.nuc_concs, train.enrichment, 
                          test.nuc_concs, test.enrichment, 
                          reg['e_l1_k'], reg['e_l2_k'], reg['e_rr_a'])
            b = regression(train.nuc_concs, train.burnup, 
                           test.nuc_concs, test.burnup, 
                           reg['b_l1_k'], reg['b_l2_k'], reg['b_rr_a'])
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
    pd.DataFrame(reactor_acc, columns=cols, index=percent_err).to_csv('reactor.csv')
    pd.DataFrame(enrichment_err, columns=cols, index=percent_err).to_csv('enrichment.csv')
    pd.DataFrame(burnup_err, columns=cols, index=percent_err).to_csv('burnup.csv')

    return 
