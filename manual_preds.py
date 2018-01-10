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

def burnup_predict(trainX, trainY, testX, testY, k, a, g, c, CV, n_trials):

    nn = KNeighborsRegressor(n_neighbors=k)
    rr = Ridge(alpha=a)
    svr = SVR(gamma=g, C=c)
    nn_err = ()
    rr_err = ()
    svr_err = ()
    #print('format is (train, test, cv) \n')
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
            elif alg == rr:
                rr.fit(trainX, trainY)
                train_pred = rr.predict(trainX)
                test_pred = rr.predict(testX)
                cv_pred = cross_val_predict(rr, trainX, trainY, cv = CV)
            else:
                svr.fit(trainX, trainY)
                train_pred = svr.predict(trainX)
                test_pred = svr.predict(testX)
                cv_pred = cross_val_predict(svr, trainX, trainY, cv = CV)
            # negative errors
            train_mape = 1 - mean_absolute_percentage_error(trainY, train_pred)
            test_mape = 1 - mean_absolute_percentage_error(testY, test_pred)
            cv_mape = 1 - mean_absolute_percentage_error(trainY, cv_pred)
            train_rmse = -1 * sqrt(mean_squared_error(trainY, train_pred))
            test_rmse = -1 * sqrt(mean_squared_error(testY, test_pred))
            cv_rmse = -1 * sqrt(mean_squared_error(trainY, cv_pred))
            #maths
            per = (train_mape, test_mape, cv_mape)
            rms = (train_rmse, test_rmse, cv_rmse)
            mape_sum = [sum(x) for x in zip(per, mape_sum)]
            rmse_sum = [sum(x) for x in zip(rms, rmse_sum)]
        mape = [x / n_trials for x in mape_sum]
        rmse = [x / n_trials for x in rmse_sum]
        if alg == nn:
            nn_err = (mape, rmse)
        elif alg == rr:
            rr_err = (mape, rmse)
        else:
            svr_err = (mape, rmse)
    
    return nn_err, rr_err, svr_err

def m_reduc(m, train_set):
    # unsplit
    train_set.nuc_concs['burnup'] = train_set.burnup
    train = train_set.nuc_concs.sample(frac=m)
    # resplit
    trainX = train.iloc[:, 0:-1]
    trainY = train.iloc[:, -1]
    return trainX, trainY

def manual_train_and_predict(train, test):
    """
    Given training and testing data, this script runs some ML algorithms
    (currently, this is nearest neighbors with 2 distance metrics and ridge
    regression) for each prediction category: reactor type, enrichment, and
    burnup

    Parameters 
    ---------- 
    
    train : group of dataframes that include training instances and the labels
    test : group of dataframes that has the testing instances and the labels

    Returns
    -------
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
    
    # no diagnostics: burnup prediction results only
    #nn, rr, svr = burnup_predict(trainX, trainY, testX, testY, k, a, g, c, CV, n_trials)
    #add print statement here
    
    #########################
    # Manual Learning Curves#
    #########################
    m_percent = np.linspace(0.15, 1.0, 20)
    nn_err = ()
    rr_err = ()
    svr_err = ()
    for m in m_percent:
        # reduce training set
        trainX, trainY = m_reduc(m, train)
        # *_err format is (mape, rmse) and each err type is [train, test, cv]
        nn_err, rr_err, svr_err = burnup_predict(trainX, trainY, testX, testY, 
                                                 k, a, g, c, CV, n_trials)
    # save Learning Curve Results
    mape_cols = ['TrainScore', 'TestScore', 'CVScore']
    rmse_cols = ['TrainNegErr', 'TestNegErr', 'CVNegErr']
    pd.DataFrame(nn_err[0], columns=mape_cols, index=m_percent).to_csv('lc_nn_mape.csv')
    pd.DataFrame(rr_err[0], columns=mape_cols, index=m_percent).to_csv('lc_rr_mape.csv')
    pd.DataFrame(svr_err[0], columns=mape_cols, index=m_percent).to_csv('lc_svr_mape.csv')
    pd.DataFrame(nn_err[1], columns=rmse_cols, index=m_percent).to_csv('lc_nn_rmse.csv')
    pd.DataFrame(rr_err[1], columns=rmse_cols, index=m_percent).to_csv('lc_rr_rmse.csv')
    pd.DataFrame(svr_err[1], columns=rmse_cols, index=m_percent).to_csv('lc_svr_rmse.csv')
    
    
    ############################
    # Manual Validation Curves #
    ############################
    k_list = np.linspace(1, 39, 20)
    alpha_list = np.logspace(-7, 3, 20)
    gammas = np.linspace(0.0005, 0.09, 20)
    c_list = np.linspace(0.0005, 0.09, 20)
    
    nn_err = ()
    rr_err = ()
    svr_err_g = ()
    for k, a, g in k_list, alpha_list, gammas:
        nn_err, rr_err, svr_err_g = burnup_predict(trainX, trainY, testX, testY, 
                                                   k, a, g, c, CV, n_trials)
    # running extra loop since SVR has two free params
    k = 1
    a = 1
    g = 0.001
    svr_err_c = ()
    for c in c_list:
        _, _, svr_err = burnup_predict(trainX, trainY, testX, testY, 
                                       k, a, g, c, CV, n_trials)
        
    # save Validation Curve Results
    pd.DataFrame(nn_err[0], columns=mape_cols, index=k_list).to_csv('vc_nn_mape.csv')
    pd.DataFrame(rr_err[0], columns=mape_cols, index=alpha_list).to_csv('vc_rr_mape.csv')
    pd.DataFrame(svr_err_g[0], columns=mape_cols, index=gammas).to_csv('vc_svr_g_mape.csv')
    pd.DataFrame(svr_err_c[0], columns=mape_cols, index=c_list).to_csv('vc_svr_c_mape.csv')
    pd.DataFrame(nn_err[1], columns=rmse_cols, index=k_list).to_csv('vc_nn_rmse.csv')
    pd.DataFrame(rr_err[1], columns=rmse_cols, index=alpha_list).to_csv('vc_rr_rmse.csv')
    pd.DataFrame(svr_err_g[1], columns=rmse_cols, index=gammas).to_csv('vc_svr_g_rmse.csv')
    pd.DataFrame(svr_err_c[1], columns=rmse_cols, index=c_list).to_csv('vc_svr_c_rmse.csv')
    
    return
