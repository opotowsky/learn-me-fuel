from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale
from math import sqrt
import numpy as np
import pandas as pd

CV = 5
n_trials = 10

def mean_absolute_percentage_error(true, pred):
    """
    Given 2 vectors of values, true and pred, calculate the MAP error

    """

    mape = np.mean(np.abs((true - pred) / true)) * 100

    return mape

def burnup_predict(trainX, trainY, testX, testY, alg):
    """
    I document my code...
    
    """

    mape_sum = [0, 0, 0]
    rmse_sum = [0, 0, 0]
    mae_sum = [0, 0, 0]
    for n in range(0, n_trials):
        #preds
        alg.fit(trainX, trainY)
        train_pred = alg.predict(trainX)
        test_pred = alg.predict(testX)
        cv_pred = cross_val_predict(alg, trainX, trainY, cv = CV)
        # negative errors
        train_mape = -1 * mean_absolute_percentage_error(trainY, train_pred)
        test_mape = -1 * mean_absolute_percentage_error(testY, test_pred)
        cv_mape = -1 * mean_absolute_percentage_error(trainY, cv_pred)
        train_rmse = -1 * sqrt(mean_squared_error(trainY, train_pred))
        test_rmse = -1 * sqrt(mean_squared_error(testY, test_pred))
        cv_rmse = -1 * sqrt(mean_squared_error(trainY, cv_pred))
        train_mae = -1 * mean_absolute_error(trainY, train_pred)
        test_mae = -1 * mean_absolute_error(testY, test_pred)
        cv_mae = -1 * mean_absolute_error(trainY, cv_pred)
        #maths
        per = (train_mape, test_mape, cv_mape)
        rms = (train_rmse, test_rmse, cv_rmse)
        mae = (train_mae, test_mae, cv_mae)
        mape_sum = [sum(x) for x in zip(per, mape_sum)]
        rmse_sum = [sum(x) for x in zip(rms, rmse_sum)]
        mae_sum = [sum(x) for x in zip(mae, mae_sum)]
    mape = [x / n_trials for x in mape_sum]
    rmse = [x / n_trials for x in rmse_sum]
    mae = [x / n_trials for x in mae_sum]
        
    return mape, rmse, mae

def m_reduc(m, train_set):
    """
    I document.

    """
    
    # unsplit
    train = pd.DataFrame(train_set.nuc_concs)
    train['burnup'] = pd.Series(train_set.burnup, index=train.index)
    reduc = train.sample(frac=m)
    # resplit
    trainX = reduc.iloc[:, 0:-1]
    trainY = reduc.iloc[:, -1]
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

    mape_cols = ['TrainPerScore', 'TestPerScore', 'CVPerScore']
    rmse_cols = ['TrainNegRMSErr', 'TestNegRMSErr', 'CVNegRMSErr']
    mae_cols = ['TrainNegMAErr', 'TestNegMAErr', 'CVNegMAErr']
    
    trainX = scale(train.nuc_concs)
    trainY = train.burnup
    testX = test.nuc_concs
    testY = test.burnup
    
    # no diagnostics: burnup prediction results only
    #for alg_type in ('nn', 'rr', 'svr'):
    #    if alg_type == 'nn':
    #        alg = KNeighborsRegressor(n_neighbors=k) 
    #    elif alg_type == 'rr':
    #        alg = Ridge(alpha=a)
    #    else:
    #        alg = SVR(gamma=g, C=c)
    #    mape, rmse, mae = burnup_predict(trainX, trainY, testX, testY, alg)
    #add print statement here
    
    #########################
    # Manual Learning Curves#
    #########################
   # m_percent = np.linspace(0.15, 1.0, 20)
   # for alg_type in ('nn', 'rr', 'svr'):
   #     map_err = []
   #     rms_err = []
   #     ma_err = []
   #     if alg_type == 'nn':
   #         alg = KNeighborsRegressor(n_neighbors=k) 
   #     elif alg_type == 'rr':
   #         alg = Ridge(alpha=a)
   #     else:
   #         alg = SVR(gamma=g, C=c)
   #     for m in m_percent:
   #         # reduce training set size
   #         trainX, trainY = m_reduc(m, train)
   #         mape, rmse, mae = burnup_predict(trainX, trainY, testX, testY, alg)
   #         map_err.append(mape)
   #         rms_err.append(rmse)
   #         ma_err.append(mae)
   #     mape = pd.DataFrame(map_err, columns=mape_cols, index=m_percent)
   #     rmse = pd.DataFrame(rms_err, columns=rmse_cols, index=m_percent)
   #     mae = pd.DataFrame(ma_err, columns=mae_cols, index=m_percent)
   #     errors = pd.concat([mape, rmse, mae], axis=1)
   #     errors.to_csv(alg_type + '_learn_manual.csv')
    
    
    ############################
    # Manual Validation Curves #
    ############################
    k_list = np.linspace(1, 39, 20)
    alpha_list = np.logspace(-7, 3, 20)
    gamma_list = np.linspace(0.0005, 0.09, 20)
    c_list = np.linspace(0.01, 100000, 20)
    
    for alg_type in ('svr_c',):#('nn', 'rr', 'svr_c', 'svr_g'):
        map_err = []
        rms_err = []
        ma_err = []
        idx = []
        if alg_type == 'svr_c':
            idx = c_list
            for c in c_list:
                alg = SVR(C=c, gamma=g)
                pt, ms, ma = burnup_predict(trainX, trainY, testX, testY, alg)
                map_err.append(pt)
                rms_err.append(ms)
                ma_err.append(ma)
        elif alg_type == 'svr_g':
            idx = gamma_list
            c = 1000
            for g in gamma_list:
                alg = SVR(C=c, gamma=g)
                pt, ms, ma = burnup_predict(trainX, trainY, testX, testY, alg)
                map_err.append(pt)
                rms_err.append(ms)
                ma_err.append(ma)
        elif alg_type == 'nn':
            idx = k_list
            for k in k_list:
                alg = KNeighborsRegressor(n_neighbors=k)
                pt, ms, ma = burnup_predict(trainX, trainY, testX, testY, alg)
                map_err.append(pt)
                rms_err.append(ms)
                ma_err.append(ma)
        elif alg_type == 'rr':
            idx = alpha_list
            for a in alpha_list:
                alg = Ridge(alpha=a)
                pt, ms, ma = burnup_predict(trainX, trainY, testX, testY, alg)
                map_err.append(pt)
                rms_err.append(ms)
                ma_err.append(ma)
        else:
            print('Derp.\n')
        mape = pd.DataFrame(map_err, columns=mape_cols, index=idx)
        rmse = pd.DataFrame(rms_err, columns=rmse_cols, index=idx)
        mae = pd.DataFrame(ma_err, columns=mae_cols, index=idx)
        errors = pd.concat([mape, rmse, mae], axis=1)
        errors.to_csv(alg_type + '_valid_manual.csv')

    return
