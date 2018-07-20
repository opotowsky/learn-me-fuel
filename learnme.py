#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs
from scipy.stats import expon, uniform

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold, RandomizedSearchCV, learning_curve

import pandas as pd
import numpy as np

def learning_curves(X, Y, knn, rr, svr, CV, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, linear, and support vector methods), varying the
    training set size for each prediction category: reactor type, cooling
    time, enrichment, and burnup

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    knn : optimized kNN learner
    rr : optimized RR learner
    svr : optimized SVR learner
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *learning_curve.csv : csv file with learning curve results for each 
                          prediction category

    """    
    
    # Note: I'm trying to avoid loops here so the code is inelegant

    trainset_frac = np.array( [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 
                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                 )
    col_names = ['AbsTrainSize', 'TrainScore', 'CV-Score']
    tsze = 'AbsTrainSize'
    tscr = 'TrainScore'
    cscr = 'CV-Score'

    # knn
    ktsize, ktrain, kcv = learning_curve(knn, X, Y, train_sizes=trainset_frac, 
                                         cv=CV, shuffle=True)
    ktrain_mean = np.mean(ktrain, axis=1)
    kcv_mean = np.mean(kcv, axis=1)
    knn_df = pd.DataFrame({tsze : ktsize, tscr : ktrain_mean, cscr : kcv_mean}, 
                           index=trainset_frac)
    knn_df['Algorithm'] = 'knn'

    # ridge
    rtsize, rtrain, rcv = learning_curve(rr, X, Y, train_sizes=trainset_frac, 
                                         cv=CV, shuffle=True)
    rtrain_mean = np.mean(rtrain, axis=1)
    rcv_mean = np.mean(rcv, axis=1)
    rr_df = pd.DataFrame({tsze : rtsize, tscr : rtrain_mean, cscr : rcv_mean}, 
                         index=trainset_frac)
    rr_df['Algorithm'] = 'rr'
    
    # svr
    stsize, strain, scv = learning_curve(svr, X, Y, train_sizes=trainset_frac, 
                                         cv=CV, shuffle=True)
    strain_mean = np.mean(strain, axis=1)
    scv_mean = np.mean(scv, axis=1)
    svr_df = pd.DataFrame({tsze : stsize, tscr : strain_mean, cscr : scv_mean}, 
                           index=trainset_frac)
    svr_df['Algorithm'] = 'svr'

    lc_data = pd.concat([knn_df, rr_df, svr_df])
    lc_data.index.name = 'TrainSizeFrac'
    lc_data.to_csv(csv_name + '_learning_curve.csv')
    return 

def track_predictions(trainX, trainY, knn_init, rr_init, svr_init, scores, CV, csv_name):
    """
    Saves csv's with predictions of each reactor parameter instance.
    
    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    knn_init : optimized kNN learner
    rr_init : optimized RR learner
    svr_init : optimized SVR learner
    scores : list of scoring types (from sckikit-learn)
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *predictions.csv : csv file with prediction results 

    """
    knn = cross_val_predict(knn_init, trainX, y=trainY, cv=CV)
    rr = cross_val_predict(rr_init, trainX, y=trainY, cv=CV)
    svr = cross_val_predict(svr_init, trainX, y=trainY, cv=CV)

    preds_by_alg = pd.DataFrame({'TrueY': trainY, 'kNN': knn, 
                                 'Ridge': rr, 'SVR': svr}, 
                                 index=trainY.index)
    preds_by_alg.to_csv(csv_name + '_predictions.csv')
    return

def errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, scores, CV, csv_name):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    knn_init : optimized kNN learner
    rr_init : optimized RR learner
    svr_init : optimized SVR learner
    scores : list of scoring types (from sckikit-learn)
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *scores.csv : csv file with scores for each CV fold
    
    """
    # init/empty the lists
    knn_scores = []
    rr_scores = []
    svr_scores = []
    for alg in ('knn', 'rr', 'svr'):
        # get init'd learner
        if alg == 'knn':
            alg_pred = knn_init
        elif alg == 'rr':
            alg_pred = rr_init
        else:
            alg_pred = svr_init
        # cross valiation to obtain scores
        cv_info = cross_validate(alg_pred, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
        df = pd.DataFrame(cv_info)
        # for concatting since it's finnicky
        if alg == 'knn':
            df['algorithm'] = 'knn'
            knn_scores = df
        elif alg == 'rr':
            df['algorithm'] = 'rr'
            rr_scores = df
        else:
            df['algorithm'] = 'svr'
            svr_scores = df
    cv_results = [knn_scores, rr_scores, svr_scores]
    df = pd.concat(cv_results)
    df.to_csv(csv_name + '_scores.csv')
    return

def main():
    """
    Given training data, this script trains and tracks each prediction for
    several algorithms and saves the predictions and ground truth to a CSV file
    """
    # Parameters for the training and predictions
    CV = 10
    
    subsets = ('fiss', 'act', 'fissact', 'all')
    subset = subsets[2]
    
    pkl_base = './pkl_trainsets/2jul2018/2jul2018_trainset'
    
    for trainset in ('1', '2'):
        pkl = pkl_base + trainset + '_nucs_' + subset + '_not-scaled.pkl'
        trainXY = pd.read_pickle(pkl)
        trainX, rY, cY, eY, bY = splitXY(trainXY)
        if subset == 'all':
            top_n = 100
            nuc_set = top_nucs(trainX, top_n)
            trainX = filter_nucs(trainX, nuc_set, top_n)
        trainX = scale(trainX)
        
        # loops through each reactor parameter to do separate predictions
        for Y in ('r', 'b', 'c', 'e'):
            trainY = pd.Series()
            # get param names and set ground truth
            if Y == 'c':
                trainY = cY
                parameter = 'cooling'
            elif Y == 'e': 
                trainY = eY
                parameter = 'enrichment'
            elif Y == 'b':
                trainY = bY
                parameter = 'burnup'
            else:
                trainY = rY
                parameter = 'reactor'
            
            #######################
            # optimize parameters #
            #######################
            
            # initialize learners
            score = 'explained_variance'
            kfold = KFold(n_splits=CV, shuffle=True)
            knn_init = KNeighborsRegressor(weights='distance')
            rr_init = Ridge()
            svr_init = SVR()
            if Y is 'r':
                score = 'accuracy'
                kfold = StratifiedKFold(n_splits=CV, shuffle=True)
                knn_init = KNeighborsClassifier(weights='distance')
                rr_init = RidgeClassifier(class_weight='balanced')
                svr_init = SVC(class_weight='balanced')
            
            # CV search the hyperparams
            knn_grid = {'n_neighbors': np.linspace(1, 50).astype(int)}
            rr_grid = {'alpha': np.logspace(-4, 10)} 
            svr_grid = {'C': expon(scale=100), 'gamma': expon(scale=.1)}
            knn_opt = RandomizedSearchCV(estimator=knn_init, param_distributions=knn_grid, 
                                         n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                         return_train_score=True)
            rr_opt = RandomizedSearchCV(estimator=rr_init, param_distributions=rr_grid,
                                         n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                         return_train_score=True)
            svr_opt = RandomizedSearchCV(estimator=svr_init, param_distributions=svr_grid,
                                         n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                         return_train_score=True)
            knn_opt.fit(trainX, trainY)
            rr_opt.fit(trainX, trainY)
            svr_opt.fit(trainX, trainY)

            ## Get best params
            #k = knn_opt.best_params_['n_neighbors']
            #a = rr_opt.best_params_['alpha']
            #g = svr_opt.best_params_['gamma']
            #c = svr_opt.best_params_['C']
            #
            ## Save dat info
            #param_file = 'trainset_' + trainset + '_hyperparameters.txt'
            #with open(param_file, 'a') as pf:
            #    pf.write('The following parameters are best from the randomized search for the {} parameter prediction:\n'.format(parameter))
            #    pf.write('k for knn is {}\n'.format(k)) 
            #    pf.write('alpha for ridge is {}\n'.format(a)) 
            #    pf.write('gamma for svr is {}\n'.format(g)) 
            #    pf.write('C for svr is {}\n'.format(c)) 
            #knn_df = pd.DataFrame(knn_opt.cv_results_)
            #rr_df = pd.DataFrame(rr_opt.cv_results_)
            #svr_df = pd.DataFrame(svr_opt.cv_results_)
            #knn_df.to_csv(param_file, sep=' ', mode='a')
            #rr_df.to_csv(param_file, sep=' ', mode='a')
            #svr_df.to_csv(param_file, sep=' ', mode='a')

            ########################
            # run predictions, etc #
            ########################
            
            # gather optimized learners
            knn_init = knn_opt.best_estimator_
            rr_init = rr_opt.best_estimator_
            svr_init = svr_opt.best_estimator_

            scores = ['explained_variance', 'neg_mean_absolute_error']
            if Y is 'r':
                scores = ['accuracy', ]
            csv_name = 'trainset_' + trainset + '_' + subset + '_' + parameter
            
            # track predictions 
            #track_predictions(trainX, trainY, knn_init, rr_init, svr_init, scores, kfold, csv_name)

            # calculate errors and scores
            #errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, scores, kfold, csv_name)

            # learning curves
            learning_curves(trainX, trainY, knn_init, rr_init, svr_init, kfold, csv_name)
            
            print("The {} predictions in trainset {} are complete\n".format(parameter, trainset), flush=True)

    return

if __name__ == "__main__":
    main()
