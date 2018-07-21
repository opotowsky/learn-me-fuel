#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs
from scipy.stats import expon, uniform

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold, RandomizedSearchCV, learning_curve

import pandas as pd
import numpy as np

def learning_curves(X, Y, alg1, alg2, alg3, CV, csv_name):
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

    # dtree
    1tsize, 1train, 1cv = learning_curve(alg1, X, Y, train_sizes=trainset_frac, 
                                         cv=CV, shuffle=True)
    1train_mean = np.mean(1train, axis=1)
    1cv_mean = np.mean(1cv, axis=1)
    1_df = pd.DataFrame({tsze : 1tsize, tscr : 1train_mean, cscr : 1cv_mean}, 
                         index=trainset_frac)
    1_df['Algorithm'] = 'dtree'

    # xtree
    2tsize, 2train, 2cv = learning_curve(alg2, X, Y, train_sizes=trainset_frac, 
                                         cv=CV, shuffle=True)
    2train_mean = np.mean(2train, axis=1)
    2cv_mean = np.mean(2cv, axis=1)
    2_df = pd.DataFrame({tsze : 2tsize, tscr : 2train_mean, cscr : 2cv_mean}, 
                         index=trainset_frac)
    2_df['Algorithm'] = 'xtree'
    
    # bayes
    3tsize, 3train, 3cv = learning_curve(alg3, X, Y, train_sizes=trainset_frac, 
                                         cv=CV, shuffle=True)
    3train_mean = np.mean(3train, axis=1)
    3cv_mean = np.mean(3cv, axis=1)
    3_df = pd.DataFrame({tsze : 3tsize, tscr : 3train_mean, cscr : 3cv_mean}, 
                         index=trainset_frac)
    3_df['Algorithm'] = 'bayes'

    lc_data = pd.concat([1_df, 2_df, 3_df])
    lc_data.index.name = 'TrainSizeFrac'
    lc_data.to_csv(csv_name + '_learning_curve.csv')
    return 

def track_predictions(trainX, trainY, alg1_init, alg2_init, alg3_init, scores, CV, csv_name):
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
    alg1 = cross_val_predict(alg1_init, trainX, y=trainY, cv=CV)
    alg2 = cross_val_predict(alg2_init, trainX, y=trainY, cv=CV)
    alg3 = cross_val_predict(alg3_init, trainX, y=trainY, cv=CV)

    preds_by_alg = pd.DataFrame({'TrueY': trainY, 'DecTree': alg1, 
                                 'ExTree': alg2, 'Bayes': alg3}, 
                                 index=trainY.index)
    preds_by_alg.to_csv(csv_name + '_predictions.csv')
    return

def errors_and_scores(trainX, trainY, alg1_init, alg2_init, alg3_init, scores, CV, csv_name):
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
    # Note: I'm trying to avoid loops here so the code is inelegant
    
    1cv_scr = cross_validate(alg1_init, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
    1_df = pd.DataFrame(1cv_scr)
    1_df['Algorithm'] = 'dtree'
    
    2cv_scr = cross_validate(alg2_init, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
    2_df = pd.DataFrame(2cv_scr)
    2_df['Algorithm'] = 'xtree'
    
    3cv_scr = cross_validate(alg3_init, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
    3_df = pd.DataFrame(3cv_scr)
    3_df['Algorithm'] = 'bayes'
    
    cv_results = [1_df, 2_df, 3_df]
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
            alg1_init = DecisionTreeRegressor()
            alg2_init = ExtraTreeRegressor()
            alg3_init = BayesianRidge()
            if Y is 'r':
                score = 'accuracy'
                kfold = StratifiedKFold(n_splits=CV, shuffle=True)
                alg1_init = DecisionTreeClassifier(weights='distance')
                alg2_init = ExtraTreeClassifier(class_weight='balanced')
                alg3_init = GaussianNB(class_weight='balanced')
            
            # CV search the hyperparams
            alg1_grid = {'n_neighbors': np.linspace(1, 50).astype(int)}
            alg2_grid = {'alpha': np.logspace(-4, 10)} 
            alg3_grid = {'C': expon(scale=100), 'gamma': expon(scale=.1)}
            alg1_opt = RandomizedSearchCV(estimator=alg1_init, param_distributions=alg1_grid, 
                                          n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                          return_train_score=True)
            alg2_opt = RandomizedSearchCV(estimator=alg2_init, param_distributions=alg2_grid,
                                          n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                          return_train_score=True)
            alg3_opt = RandomizedSearchCV(estimator=alg3_init, param_distributions=alg3_grid,
                                          n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                          return_train_score=True)
            alg1_opt.fit(trainX, trainY)
            alg2_opt.fit(trainX, trainY)
            alg3_opt.fit(trainX, trainY)

            # Get best params
            p1 = alg1_opt.best_params_['n_neighbors']
            p2 = alg2_opt.best_params_['alpha']
            p3 = alg3_opt.best_params_['gamma']
            p4 = alg3_opt.best_params_['C']
            
            # Save dat info
            param_file = 'trainset_' + trainset + '_hyperparameters_alt-algs.txt'
            with open(param_file, 'a') as pf:
                pf.write('The following parameters are best from the randomized search for the {} parameter prediction:\n'.format(parameter))
                pf.write('k for knn is {}\n'.format(p1)) 
                pf.write('alpha for ridge is {}\n'.format(p2)) 
                pf.write('gamma for svr is {}\n'.format(p3)) 
                pf.write('C for svr is {}\n'.format(p4)) 
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
            alg1_init = alg1_opt.best_estimator_
            alg2_init = alg2_opt.best_estimator_
            alg3_init = alg3_opt.best_estimator_

            scores = ['explained_variance', 'neg_mean_absolute_error']
            if Y is 'r':
                scores = ['accuracy', ]
            csv_name = 'trainset_' + trainset + '_' + subset + '_' + parameter
            
            print("The {} predictions in trainset {} are beginning\n".format(parameter, trainset), flush=True)
            
            # track predictions 
            track_predictions(trainX, trainY, alg1_init, alg2_init, alg3_init, scores, kfold, csv_name)
            print("\t Prediction tracking done\n", flush=True)

            # calculate errors and scores
            errors_and_scores(trainX, trainY, alg1_init, alg2_init, alg3_init, scores, kfold, csv_name)
            print("\t CV scoring done\n", flush=True)

            # learning curves
            learning_curves(trainX, trainY, alg1_init, alg2_init, alg3_init, kfold, csv_name)
            print("\t Learning curves done\n", flush=True)
            
            print("The {} predictions in trainset {} are complete\n".format(parameter, trainset), flush=True)

    return

if __name__ == "__main__":
    main()
