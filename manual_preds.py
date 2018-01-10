from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
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

def regression(trainX, trainY, testX, testY, k, a, g, c):
    """ 
    Training for Regression: predicts a model using three algorithms, returning
    the training error, testing error, and cross validation error for each. 

    """
    
    l2 = KNeighborsRegressor(n_neighbors = k, metric='l2', p=2)
    l2.fit(trainX, trainY)
    
    rr = Ridge(alpha = a)
    rr.fit(trainX, trainY)
    
    svr = SVR(C=c, gamma=g)
    svr.fit(trainX, trainY)
    svr_train_pred = svr.predict(trainX)
    svr_test_pred = svr.predict(testX)
    svr_cv_pred = cross_val_predict(svr, trainX, trainY, cv = 5)
    
    svr_train_mape = mean_absolute_percentage_error(trainY, train_pred)
    svr_test_mape = mean_absolute_percentage_error(testY, test_pred)
    svr_cv_mape = mean_absolute_percentage_error(trainY, cv_pred)
    svr_train_rmse = mean_absolute_percentage_error(trainY, train_pred)
    svr_test_rmse = mean_absolute_percentage_error(testY, test_pred)
    svr_cv_rmse = mean_absolute_percentage_error(trainY, cv_pred)
    
    return (train_mape, test_mape, cv_mape)

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
    #train_mape1 = mean_absolute_percentage_error(trainY, train_predict1)
    #train_mape2 = mean_absolute_percentage_error(trainY, train_predict2)
    #train_mape3 = mean_absolute_percentage_error(trainY, train_predict3)
    #test_mape1 = mean_absolute_percentage_error(testY, test_predict1)
    #test_mape2 = mean_absolute_percentage_error(testY, test_predict2)
    #test_mape3 = mean_absolute_percentage_error(testY, test_predict3)
    #cv_mape1 = mean_absolute_percentage_error(trainY, cv_predict1)
    #cv_mape2 = mean_absolute_percentage_error(trainY, cv_predict2)
    #cv_mape3 = mean_absolute_percentage_error(trainY, cv_predict3)
    #rmse = (train_mape1, train_mape2, train_mape3, test_mape1, test_mape2, test_mape3, 
    #        cv_mape1, cv_mape2, cv_mape3)

    #return rmse

def manual_train_and_predict(train, test):
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

    k = 1
    a = 1
    g = 0.001
    c = 1000
    CV = 5
    n_trials = 10

    trainX = train.nuc_concs
    trainY = train.burnup
    testX = test.nuc_concs
    testY = test.burnup

    #################
    ## Results Only##
    #################
    nn = KNeighborsRegressor(n_neighbors=k)
    rr = Ridge(alpha=a)
    svr = SVR(gamma=g, C=c)
    print('format is (train, test, cv) \n')
    for alg in (nn, rr, svr):
        mape_sum = (0, 0, 0)
        rmse_sum = (0, 0, 0)
        for n in range(0, n_trials):
            #preds`
            if alg == nn:
                nn.fit(trainX, trainY)
                train_pred = nn.predict(trainX)
                test_pred = nn.predict(testX)
                cv_pred = cross_val_predict(nn, trainX, trainY, cv = CV)
                print('NN errors are as follows: \n')
            elif alg == rr:
                rr.fit(trainX, trainY)
                train_pred = rr.predict(trainX)
                test_pred = rr.predict(testX)
                cv_pred = cross_val_predict(rr, trainX, trainY, cv = CV)
                print('RR errors are as follows: \n')
            else:
                svr.fit(trainX, trainY)
                train_pred = svr.predict(trainX)
                test_pred = svr.predict(testX)
                cv_pred = cross_val_predict(svr, trainX, trainY, cv = CV)
                print('SVRs errors are as follows: \n')
            #errs
            train_mape = mean_absolute_percentage_error(trainY, train_pred)
            test_mape = mean_absolute_percentage_error(testY, test_pred)
            cv_mape = mean_absolute_percentage_error(trainY, cv_pred)
            train_rmse = sqrt(mean_squared_error(trainY, train_pred))
            test_rmse = sqrt(mean_squared_error(testY, test_pred))
            cv_rmse = sqrt(mean_squared_error(trainY, cv_pred))
            #maths
            per = (train_mape, test_mape, cv_mape)
            rms = (train_rmse, test_rmse, cv_rmse)
            mape_sum = [sum(x) for x in zip(per, mape_sum)]
            rmse_sum = [sum(x) for x in zip(rms, rmse_sum)]
        mape = [x / n_trials for x in mape_sum]
        rmse = [x / n_trials for x in rmse_sum]
        print('MAPE: ')
        print(mape)
        print('\n')
        print('RMSE: ')
        print(rmse)
        print('\n')
    
    
    #################
    ##Manual Curves##
    #################

    #m_percent = np.linspace(0.15, 1.0, 20)
    #k_list = np.linspace(1, 39, 20)
    #alpha_list = np.logspace(-10, 4, 15)
    #gammas = np.linspace(0.0005, 0.09, 20)
    #c_list = np.linspace(0.0005, 0.09, 20)
    #
    #n_trials = 1 
    #burnup_err = []
    #idx = []
    #for m in m_percent:
    #    b_sum = (0, 0, 0)
    #    for i in range(0, n_trials):
    #        #burnup = regression(trainX, trainY, testX, testY, k, a, g, c)
    #        b_sum = [sum(x) for x in zip(b, b_sum)]
    #    burnup = [x / n_trials for x in b_sum]
    #    burnup_err.append(burnup)
    #    idx.append(n_sample)
    #
    #train = []
    #test = []
    #cv = []
    #for g in gammas:
    #    svr = SVR(C=C, gamma=g)
    #    svr.fit(trainX, trainY)
    #    train_pred = svr.predict(trainX)
    #    test_pred = svr.predict(testX)
    #    cv_pred = cross_val_predict(svr, trainX, trainY, cv = 5)
    #    train_mape = mean_absolute_percentage_error(trainY, train_pred)
    #    test_mape = mean_absolute_percentage_error(testY, test_pred)
    #    cv_mape = mean_absolute_percentage_error(trainY, cv_pred)
    #    train.append(train_mape)
    #    test.append(test_mape)
    #    cv.append(cv_mape)

    #
    #
    ##Save results
    #cols = ['TrainL1', 'TrainL2', 'TrainRR', 'TestL1', 'TestL2', 'TestRR', 'CVL1', 'CVL2', 'CVRR']
    #cols = ['TrainErr', 'TestErr', 'CVErr']
    #pd.DataFrame(burnup_err, columns = cols, index = idx).to_csv('svrburnup.csv')
    #pd.DataFrame({'Gammas': gammas, 'TrainErr': train, 'TestErr': test, 'CVErr': cv}).to_csv('svrgammas.csv')
    
    return
