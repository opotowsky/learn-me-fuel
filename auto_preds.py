from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import numpy as np
import pandas as pd

CV = 10
m_percent = np.linspace(0.15, 1.0, 20)

def diagnostic_curves(trainX, trainY, alg1, alg2, param_string, param_list, alg_type):
    """
    Document me

    """

    scores = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    lc_errors = pd.DataFrame() 
    vc_errors= pd.DataFrame()
    alg1.fit(trainX, trainY)
    for score in scores:
        if param_string != 'C':
            # Don't need extra SVR LC since they are based on same defaults
            m, lct, lcv = learning_curve(alg1, trainX, trainY, cv=CV, 
                                         train_sizes=m_percent, scoring=score)
            lct_data = pd.Series(np.mean(lct, axis=1), index=m_percent, name=score+'_LC-Train')
            lcv_data = pd.Series(np.mean(lcv, axis=1), index=m_percent, name=score+'_LC-CV')
            lc_errors = pd.concat([lc_errors, lct_data, lcv_data], axis=1)
        vct, vcv = validation_curve(alg2, trainX, trainY, param_string, 
                                    param_list, cv=CV, scoring=score)
        vct_data = pd.Series(np.mean(vct, axis=1), index=param_list, name=score+'_VC-Train')
        vcv_data = pd.Series(np.mean(vcv, axis=1), index=param_list, name=score+'_VC-CV')
        vc_errors = pd.concat([vc_errors, vct_data, vcv_data], axis=1)
    if param_string != 'C':
        lc_errors.to_csv(alg_type + '_learn_auto.csv')
    vc_errors.to_csv(alg_type + '_valid_auto.csv')

    return

def auto_train_and_predict(train):
    """
    Given training and testing data, this script runs some ML algorithms
    to predict burnup

    Parameters 
    ---------- 
    
    train : group of dataframes that include training data and the three
            labels 
    
    Returns
    -------
    burnup : tuples of error metrics for training, testing, and cross validation 
             errors for all three algorithms

    """

    # some params
    k = 1
    a = 1
    g = 0.001
    c = 1000
    k_list = np.linspace(1, 39, 20)
    alpha_list = np.logspace(-7, 3, 20)
    gamma_list = np.linspace(0.0005, 0.09, 20)
    c_list = np.linspace(0.0005, 0.09, 20)

    trainX = train.nuc_concs
    trainY = train.burnup
    
    for alg_type in ('svr_c', 'svr_g'):#('nn', 'rr', 'svr_c', 'svr_g'):
        if alg_type == 'nn':
            alg1 = KNeighborsRegressor(n_neighbors=k)
            alg2 = KNeighborsRegressor()
            param_string = 'n_neighbors'
            param_list = k_list
        elif alg_type == 'svr_c':
            alg1 = SVR(gamma=g, C=c)
            alg2 = SVR(gamma=g)
            param_string = 'C'
            param_list = c_list
        elif alg_type == 'svr_g':
            alg1 = SVR(gamma=g, C=c)
            alg2 = SVR(C=c)
            param_string = 'gamma'
            param_list = gamma_list
        else:
            alg1 = Ridge(alpha=a)
            alg2 = Ridge()
            param_string = 'alpha'
            param_list = alpha_list            
        diagnostic_curves(trainX, trainY, alg1, alg2, param_string, param_list, alg_type)
    
    return
