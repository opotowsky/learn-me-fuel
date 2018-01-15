from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import numpy as np
import pandas as pd

# some defaults
k = 1
a = 1
g = 0.01
c = 10
CV = 5
n_trials = 10
nn = KNeighborsRegressor(n_neighbors=k)
rr = Ridge(alpha=a)
svr = SVR(gamma=g, C=c)

def pred_errs(trainX, trainY, testX, testY, alg):
    """
    Document me

    """

    mape_sum = [0, 0, 0]
    rmse_sum = [0, 0, 0]
    mae_sum = [0, 0, 0]
    for n in range(0, n_trials):
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

def mean_absolute_percentage_error(true, pred):
    """
    Given 2 vectors of values, true and pred, calculate the MAP error

    """

    mape = np.mean(np.abs((true - pred) / true)) * 100

    return mape

def add_error(percent_err, df):
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
    x = df.shape[0]
    y = df.shape[1]
    err = percent_err / 100.0
    low = 1 - err
    high = 1 + err
    errs = np.random.uniform(low, high, (x, y))
    df_err = df * errs
    
    return df_err

def random_error(train, test):
    """
    Given training and testing data, this script runs some ML algorithms
    to predict burnup in the face of reduced information

    Parameters 
    ---------- 
    
    train : group of dataframes that include training instances and the labels
    test : group of dataframes that has the testing instances and the labels

    Returns
    -------
    burnup : tuple of different error metrics for training, testing, and cross validation 
             errors for all three algorithms

    """
    
    mape_cols = ['TrainNegPercent', 'TestNegPercent', 'CVNegPercent']
    rmse_cols = ['TrainNegRMSErr', 'TestNegRMSErr', 'CVNegRMSErr']
    mae_cols = ['TrainNegMAErr', 'TestNegMAErr', 'CVNegMAErr']

    trainX = train.nuc_concs
    trainY = train.burnup
    testX = test.nuc_concs
    testY = test.burnup
    
    err_percent = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.1, 1.4, 
                   1.7, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9]
    for alg in (svr,):#(nn, rr, svr):
        map_err = []
        rms_err = []
        ma_err = []
        for err in err_percent:
            trainX = add_error(err, trainX)
            # *_err format for each err type is [train, test, cv]
            mape, rmse, mae = pred_errs(trainX, trainY, testX, testY,  alg)
            map_err.append(mape)
            rms_err.append(rmse)
            ma_err.append(mae)
        mape = pd.DataFrame(map_err, columns=mape_cols, index=err_percent)
        rmse = pd.DataFrame(rms_err, columns=rmse_cols, index=err_percent)
        mae = pd.DataFrame(ma_err, columns=mae_cols, index=err_percent)
        errors = pd.concat([mape, rmse, mae], axis=1)
        if alg == nn:
            errors.to_csv('nn_inforeduc.csv')
        elif alg == rr:
            errors.to_csv('rr_inforeduc.csv')
        else:
            errors.to_csv('svr_inforeduc_g01c10.csv')
    
    return
