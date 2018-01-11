from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
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
    
    ## L1 norm is Manhattan Distance
    ## L2 norm is Euclidian Distance 
    ## Ridge Regression is Linear + L2 regularization
    #l1 = KNeighborsRegressor(n_neighbors = k_l1, metric='l1', p=1)
    #l2 = KNeighborsRegressor(n_neighbors = k_l2, metric='l2', p=2)
    #rr = Ridge(alpha = a_rr)
    #l1.fit(trainX, trainY)
    #l2.fit(trainX, trainY)
    #rr.fit(trainX, trainY)
    
    # Testing SVR
    C = 100.0
    gamma = 0.001
    svr = SVR(C=C, gamma=gamma)
    svr.fit(trainX, trainY)
    train_pred = svr.predict(trainX)
    test_pred = svr.predict(testX)
    cv_pred = cross_val_predict(svr, trainX, trainY, cv = 5)
    train_err = mean_absolute_percentage_error(trainY, train_pred)
    test_err = mean_absolute_percentage_error(testY, test_pred)
    cv_err = mean_absolute_percentage_error(trainY, cv_pred)
    
    return (train_err, test_err, cv_err)

    ## Predictions for training, testing, and cross validation 
    #train_predict1 = l1.predict(trainX)
    #train_predict2 = l2.predict(trainX)
    #train_predict3 = rr.predict(trainX)
    #test_predict1 = l1.predict(testX)
    #test_predict2 = l2.predict(testX)
    #test_predict3 = rr.predict(testX)
    #cv_predict1 = cross_val_predict(l1, trainX, trainY, cv = 5)
    #cv_predict2 = cross_val_predict(l2, trainX, trainY, cv = 5)
    #cv_predict3 = cross_val_predict(rr, trainX, trainY, cv = 5)
    #train_err1 = mean_absolute_percentage_error(trainY, train_predict1)
    #train_err2 = mean_absolute_percentage_error(trainY, train_predict2)
    #train_err3 = mean_absolute_percentage_error(trainY, train_predict3)
    #test_err1 = mean_absolute_percentage_error(testY, test_predict1)
    #test_err2 = mean_absolute_percentage_error(testY, test_predict2)
    #test_err3 = mean_absolute_percentage_error(testY, test_predict3)
    #cv_err1 = mean_absolute_percentage_error(trainY, cv_predict1)
    #cv_err2 = mean_absolute_percentage_error(trainY, cv_predict2)
    #cv_err3 = mean_absolute_percentage_error(trainY, cv_predict3)
    #rmse = (train_err1, train_err2, train_err3, test_err1, test_err2, test_err3, 
    #        cv_err1, cv_err2, cv_err3)

    #return rmse

def auto_train_and_predict(train, test):
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

    k = 3
    a = 100
    gamma = 0.01
    C = 1

    burnup = regression(train.nuc_concs, train.burnup, 
                        test.nuc_concs, test.burnup, 
                        k, a, gamma, C)
    
    
    m_size = np.linspace(0.15, 1.0, 20)
    gammas = np.linspace(0.0005, 0.09, 20)
    svr_train_sizes, svr_train_scores, svr_valid_scores = learning_curve(svr, trainX, trainYb, cv=5, train_sizes=m_size)
    svr_train_scores, svr_valid_scores = validation_curve(svr, trainX, trainYb, "gamma", gammas, cv=5)
    svr_train_mean = np.mean(svr_train_scores, axis=1)
    svr_valid_mean = np.mean(svr_valid_scores, axis=1)
    pd.DataFrame({'TrainSize': svr_train_sizes, 'TrainScore': svr_train_mean, 'ValidScore': svr_valid_mean}).to_csv('svrlearn.csv')
    pd.DataFrame({'Gammas': gammas, 'TrainScore': svr_train_mean, 'ValidScore': svr_valid_mean}).to_csv('svrvalid.csv')
    
    
    rr = Ridge(alpha=a)
    alpha_list = np.logspace(-10, 4, 15)
    rrcv = RidgeCV(alphas=alpha_list) ###
    rrcv.fit(trainX, trainYb)
    rr_train_sizes, rr_train_scores, rr_valid_scores = learning_curve(rr, trainX, trainYb, cv=5, train_sizes=m_size)
    rr_train_scores, rr_valid_scores = validation_curve(Ridge(), trainX, trainYb, "alpha", alphas, cv=5)
    rr_train_mean = np.mean(rr_train_scores, axis=1)
    rr_valid_mean = np.mean(rr_valid_scores, axis=1)
    pd.DataFrame({'TrainSize': rr_train_sizes, 'TrainScore': rr_train_mean, 'ValidScore': rr_valid_mean}).to_csv('ridgelearn.csv')
    pd.DataFrame({'Alpha': alphas, 'TrainScore': rr_train_mean, 'ValidScore': rr_valid_mean}).to_csv('ridgevalid.csv')
    
    
    
    return burnup
