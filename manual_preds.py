from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
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

    ## regularization parameters (k and alpha (a) differ for each)
    #reg = {'r_l1_k' : 15, 'r_l2_k' : 20, 'r_rc_a' : 0.1, 
    #       'e_l1_k' : 7, 'e_l2_k' : 10, 'e_rr_a' : 100, 
    #       'b_l1_k' : 30, 'b_l2_k' : 35, 'b_rr_a' : 100}

    #reactor = classification(train.nuc_concs, train.reactor, 
    #                         test.nuc_concs, test.reactor, 
    #                         reg['r_l1_k'], reg['r_l2_k'], reg['r_rc_a'])
    #enrichment = regression(train.nuc_concs, train.enrichment, 
    #                        test.nuc_concs, test.enrichment, 
    #                        reg['e_l1_k'], reg['e_l2_k'], reg['e_rr_a'])
    #burnup = regression(train.nuc_concs, train.burnup, 
    #                    test.nuc_concs, test.burnup, 
    #                    reg['b_l1_k'], reg['b_l2_k'], reg['b_rr_a'])
    burnup = regression(train.nuc_concs, train.burnup, 
                        test.nuc_concs, test.burnup, 
                        30, 35, 300000)
    
    #################################
    ########## SVR Stuff ############
    #################################
    # Step 1, find ideal params
    #C_range = [10000, 50000, 100000] #np.logspace(1, 4, 4)
    #gamma_range = [0.01, 0.001, 0.0001] #np.logspace(-6, -3, 4)
    #param_grid = dict(gamma=gamma_range, C=C_range)
    ##cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    #grid = GridSearchCV(SVR(), param_grid=param_grid)
    #grid.fit(trainX, trainYb)
    #print("The best parameters are %s with a score of %0.2f" 
    #      % (grid.best_params_, grid.best_score_))
    # Step 2, stock learning/valid curves
    #C = 100.0
    #gamma = 0.001
    #partial = np.linspace(0.15, 1.0, 20)
    #gammas = np.linspace(0.0005, 0.09, 20)
    #svr_train_sizes, svr_train_scores, svr_valid_scores = learning_curve(svr, trainX, trainYb, cv=5, train_sizes=partial)
    #svr_train_scores, svr_valid_scores = validation_curve(svr, trainX, trainYb, "gamma", gammas, cv=5)
    #svr_train_mean = np.mean(svr_train_scores, axis=1)
    #svr_valid_mean = np.mean(svr_valid_scores, axis=1)
    #pd.DataFrame({'TrainSize': svr_train_sizes, 'TrainScore': svr_train_mean, 'ValidScore': svr_valid_mean}).to_csv('svrlearn.csv')
    #pd.DataFrame({'Gammas': gammas, 'TrainScore': svr_train_mean, 'ValidScore': svr_valid_mean}).to_csv('svrvalid.csv')
    # 
    # Step 3, manual gammas validation
   # train = []
   # test = []
   # cv = []
   # for g in gammas:
   #     svr = SVR(C=C, gamma=g)
   #     svr.fit(trainX, trainYb)
   #     train_pred = svr.predict(trainX)
   #     test_pred = svr.predict(testX)
   #     cv_pred = cross_val_predict(svr, trainX, trainYb, cv = 5)
   #     train_err = mean_absolute_percentage_error(trainYb, train_pred)
   #     test_err = mean_absolute_percentage_error(testYb, test_pred)
   #     cv_err = mean_absolute_percentage_error(trainYb, cv_pred)
   #     train.append(train_err)
   #     test.append(test_err)
   #     cv.append(cv_err)

   # pd.DataFrame({'Gammas': gammas, 'TrainErr': train, 'TestErr': test, 'CVErr': cv}).to_csv('svrgammas.csv')

    svr = SVR()
    svr.fit(trainX, trainYb)
    train_pred = svr.predict(trainX)
    test_pred = svr.predict(testX)
    cv_pred = cross_val_predict(svr, trainX, trainYb, cv = 5)
    train_err = mean_absolute_percentage_error(trainYb, train_pred)
    test_err = mean_absolute_percentage_error(testYb, test_pred)
    cv_err = mean_absolute_percentage_error(trainYb, cv_pred)
    train_mse = mean_squared_error(trainYb, train_pred)
    test_mse = mean_squared_error(testYb, test_pred)
    cv_mse = mean_squared_error(trainYb, cv_pred)
    svr_mape = (train_err, test_err, cv_err)
    svr_rmse = map(lambda x: math.sqrt(x), (train_mse, test_mse, cv_mse))
  
    rr = Ridge()
    rr.fit(trainX, trainYb)
    train_pred = rr.predict(trainX)
    test_pred = rr.predict(testX)
    cv_pred = cross_val_predict(rr, trainX, trainYb, cv = 5)
    train_err = mean_absolute_percentage_error(trainYb, train_pred)
    test_err = mean_absolute_percentage_error(testYb, test_pred)
    cv_err = mean_absolute_percentage_error(trainYb, cv_pred)
    train_mse = mean_squared_error(trainYb, train_pred)
    test_mse = mean_squared_error(testYb, test_pred)
    cv_mse = mean_squared_error(trainYb, cv_pred)
    rr_mape = (train_err, test_err, cv_err)
    rr_rmse = map(lambda x: math.sqrt(x), (train_mse, test_mse, cv_mse))

    nn = KNeighborsRegressor(n_neighbors=1)
    nn.fit(trainX, trainYb)
    train_pred = nn.predict(trainX)
    test_pred = nn.predict(testX)
    cv_pred = cross_val_predict(nn, trainX, trainYb, cv = 5)
    train_err = mean_absolute_percentage_error(trainYb, train_pred)
    test_err = mean_absolute_percentage_error(testYb, test_pred)
    cv_err = mean_absolute_percentage_error(trainYb, cv_pred)
    train_mse = mean_squared_error(trainYb, train_pred)
    test_mse = mean_squared_error(testYb, test_pred)
    cv_mse = mean_squared_error(trainYb, cv_pred)
    nn_mape = (train_err, test_err, cv_err)
    nn_rmse = map(lambda x: math.sqrt(x), (train_mse, test_mse, cv_mse))

    print('format is train, test, cv \n')
    print('NN MAPEs are as follows \n')
    print(nn_mape)
    print('\n')
    print('NN RMSEs are as follows \n')
    print(nn_rmse)
    print('\n')
    print('RR MAPEs are as follows \n')
    print(rr_mape)
    print('\n')
    print('RR RMSEs are as follows \n')
    print(rr_rmse)
    print('\n')
    print('SVR MAPEs are as follows \n')
    print(svr_mape)
    print('\n')
    print('SVR RMSEs are as follows \n')
    print(svr_rmse)
    print('\n')

    ################################
    ####### Ridge and Lasso ########
    ################################
    #rr = Ridge(alpha=300000)
    #lr = Lasso(alpha=3000)
    #partial = np.linspace(0.15, 1.0, 20)
    #alpha_list = np.logspace(-10, 4, 15)
    #rrcv = RidgeCV(alphas=alpha_list)
    #rrcv.fit(trainX, trainYb)
    #please_work = rrcv.alpha_
    #print('Alpha:')
    #print(please_work)
    #rr_train_sizes, rr_train_scores, rr_valid_scores = learning_curve(rr, trainX, trainYb, cv=5, train_sizes=partial)
    #lr_train_sizes, lr_train_scores, lr_valid_scores = learning_curve(lr, trainX, trainYb, cv=5, train_sizes=partial)
    #rr_train_scores, rr_valid_scores = validation_curve(Ridge(), trainX, trainYb, "alpha", alphas, cv=5)
    #lr_train_scores, lr_valid_scores = validation_curve(Lasso(), trainX, trainYb, "alpha", alphas, cv=5)
    #rr_train_mean = np.mean(rr_train_scores, axis=1)
    #rr_valid_mean = np.mean(rr_valid_scores, axis=1)
    #lr_train_mean = np.mean(lr_train_scores, axis=1)
    #lr_valid_mean = np.mean(lr_valid_scores, axis=1)
    #pd.DataFrame({'TrainSize': rr_train_sizes, 'TrainScore': rr_train_mean, 'ValidScore': rr_valid_mean}).to_csv('ridgelearn.csv')
    #pd.DataFrame({'TrainSize': lr_train_sizes, 'TrainScore': lr_train_mean, 'ValidScore': lr_valid_mean}).to_csv('lassolearn.csv')
    #pd.DataFrame({'Alpha': alphas, 'TrainScore': rr_train_mean, 'ValidScore': rr_valid_mean}).to_csv('ridgevalid.csv')
    #pd.DataFrame({'Alpha': alphas, 'TrainScore': lr_train_mean, 'ValidScore': lr_valid_mean}).to_csv('lassovalid.csv')
    
    
    ###################################
    ########## Manual Stuff ###########
    ###################################
    ## Cut training set to create a learning curve
    #n_trials = 1 
    #percent_set = np.linspace(0.15, 1.0, 20)
    ##reactor_acc = []
    ##enrichment_err = []
    #burnup_err = []
    #idx = []
    #for per in percent_set:
    #    # weak point in code
    #    #r_sum = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    #    #e_sum = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    #    #b_sum = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    #    b_sum = (0, 0, 0)
    #    for i in range(0, n_trials):
    #        # could use frac in df.sample too, but think I want to see absolute 
    #        # training set size instead in the future
    #        n_sample = int(per * len(trainXY))
    #        subXY = trainXY.sample(n = n_sample)
    #        trainX, trainYr, trainYe, trainYb = splitXY(subXY)
    #        trainX = filter_nucs(trainX, nuc_set, top_n)
    #        subset = LearnSet(nuc_concs = trainX, reactor = trainYr, 
    #                          enrichment = trainYe, burnup = trainYb)
    #        #r, e, b = train_and_predict(subset, test_set)
    #        b = train_and_predict(subset, test_set)
    #        #r_sum = [sum(x) for x in zip(r, r_sum)]
    #        #e_sum = [sum(x) for x in zip(e, e_sum)]
    #        b_sum = [sum(x) for x in zip(b, b_sum)]
    #    #reactor = [x / n_trials for x in r_sum]
    #    #enrichment = [x / n_trials for x in e_sum]
    #    burnup = [x / n_trials for x in b_sum]
    #    #reactor_acc.append(reactor)
    #    #enrichment_err.append(enrichment)
    #    burnup_err.append(burnup)
    #    idx.append(n_sample)
    
    
    # Save results
    #cols = ['TrainL1', 'TrainL2', 'TrainRR', 'TestL1', 'TestL2', 'TestRR', 'CVL1', 'CVL2', 'CVRR']
    #pd.DataFrame(reactor_acc, columns = cols, index = idx).to_csv('reactor.csv')
    #pd.DataFrame(enrichment_err, columns = cols, index = idx).to_csv('enrichment.csv')
    #cols = ['ANNTrainErr', 'ANNTestErr']
    #cols = ['TrainErr', 'TestErr', 'CVErr']
    #pd.DataFrame(burnup_err, columns = cols, index = idx).to_csv('svrburnup.csv')
    
    return burnup
