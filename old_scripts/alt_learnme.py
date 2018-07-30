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
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
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
                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] )
    col_names = ['AbsTrainSize', 'TrainScore', 'CV-Score']
    tsze = 'AbsTrainSize'
    tscr = 'TrainScore'
    cscr = 'CV-Score'

    # dtree
    tsize, train, cv = learning_curve(alg1, X, Y, train_sizes=trainset_frac, 
                                      cv=CV, shuffle=True)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df1 = pd.DataFrame({tsze : tsize, tscr : train_mean, cscr : cv_mean}, 
                        index=trainset_frac)
    df1['Algorithm'] = 'dtree'

    # xtree
    tsize, train, cv = learning_curve(alg2, X, Y, train_sizes=trainset_frac, 
                                      cv=CV, shuffle=True)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df2 = pd.DataFrame({tsze : tsize, tscr : train_mean, cscr : cv_mean}, 
                        index=trainset_frac)
    df2['Algorithm'] = 'xtree'
    
    # bayes
    tsize, train, cv = learning_curve(alg3, X, Y, train_sizes=trainset_frac, 
                                      cv=CV, shuffle=True)
    train_mean = np.mean(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    df3 = pd.DataFrame({tsze : tsize, tscr : train_mean, cscr : cv_mean}, 
                        index=trainset_frac)
    df3['Algorithm'] = 'bayes'

    lc_data = pd.concat([df1, df2, df3])
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
    alg1_init : optimized learner 1
    alg2_init : optimized learner 2
    alg3_init : optimized learner 3
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
    alg1_init : optimized learner 1
    alg2_init : optimized learner 2
    alg3_init : optimized learner 3
    scores : list of scoring types (from sckikit-learn)
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *scores.csv : csv file with scores for each CV fold
    
    """
    # Note: I'm trying to avoid loops here so the code is inelegant
    
    cv_scr = cross_validate(alg1_init, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
    df1 = pd.DataFrame(cv_scr)
    df1['Algorithm'] = 'dtree'
    
    cv_scr = cross_validate(alg2_init, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
    df2 = pd.DataFrame(cv_scr)
    df2['Algorithm'] = 'xtree'
    
    cv_scr = cross_validate(alg3_init, trainX, trainY, scoring=scores, cv=CV, return_train_score=False)
    df3 = pd.DataFrame(cv_scr)
    df3['Algorithm'] = 'bayes'
    
    cv_results = [df1, df2, df3]
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
                alg1_init = DecisionTreeClassifier(class_weight='balanced')
                alg2_init = ExtraTreeClassifier(class_weight='balanced')
                alg3_init = GaussianNB()
            
            # CV search the hyperparams
            # alg1
            alg1_grid = {"max_depth": np.linspace(3, 90).astype(int), 
                         "max_features": np.linspace(5, len(trainXY.columns)-6).astype(int)}
            alg1_opt = RandomizedSearchCV(estimator=alg1_init, param_distributions=alg1_grid, 
                                          n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                          return_train_score=True)
            alg1_opt.fit(trainX, trainY)
            alg1_init = alg1_opt.best_estimator_
            d1 = alg1_opt.best_params_['max_depth']
            f1 = alg1_opt.best_params_['max_features']
            
            # alg2
            alg2_grid = alg1_grid
            alg2_opt = RandomizedSearchCV(estimator=alg2_init, param_distributions=alg2_grid,
                                          n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                          return_train_score=True)
            alg2_opt.fit(trainX, trainY)
            alg2_init = alg2_opt.best_estimator_
            d2 = alg2_opt.best_params_['max_depth']
            f2 = alg2_opt.best_params_['max_features']
            
            # alg3
            alg3_grid = {'n_iter': np.linspace(50, 1000).astype(int), 
                         'alpha_1': np.logspace(-8, 2), 'alpha_2' : np.logspace(-8, 2), 
                         'lambda_1': np.logspace(-8, 2), 'lambda_2' : np.logspace(-8, 2)}
            if Y is not 'r':
                alg3_opt = RandomizedSearchCV(estimator=alg3_init, param_distributions=alg3_grid,
                                              n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                              return_train_score=True)
                alg3_opt.fit(trainX, trainY)
                alg3_init = alg3_opt.best_estimator_
                it = alg3_opt.best_params_['n_iter']
                a1 = alg3_opt.best_params_['alpha_1']
                a2 = alg3_opt.best_params_['alpha_2']
                l1 = alg3_opt.best_params_['lambda_1']
                l2 = alg3_opt.best_params_['lambda_2']

            # Save dat info
            param_file = 'trainset_' + trainset + '_hyperparameters_alt-algs.txt'
            with open(param_file, 'a') as pf:
                pf.write('The following parameters are best from the randomized search for the {} parameter prediction:\n'.format(parameter))
                pf.write('max depth for dtree is {}\n'.format(d1)) 
                pf.write('max features for dtree is {}\n'.format(f1)) 
                pf.write('max depth for xtree is {}\n'.format(d2)) 
                pf.write('max features for xtree is {}\n'.format(f2)) 
                if Y is not 'r':
                    pf.write('num iterations for bayes reg is {}\n'.format(it))
                    pf.write('alpha 1 for bayes reg is {}\n'.format(a1))
                    pf.write('alpha 2 for bayes reg is {}\n'.format(a2))
                    pf.write('lambda 1 for bayes reg is {}\n'.format(l1))
                    pf.write('lambda 2 for bayes reg is {}\n'.format(l2))

            ########################
            # run predictions, etc #
            ########################
            
            #scores = ['explained_variance', 'neg_mean_absolute_error']
            #if Y is 'r':
            #    scores = ['accuracy', ]
            #csv_name = 'trainset_' + trainset + '_' + subset + '_' + parameter
            #
            #print("The {} predictions in trainset {} are beginning\n".format(parameter, trainset), flush=True)
            #
            ## track predictions 
            #track_predictions(trainX, trainY, alg1_init, alg2_init, alg3_init, scores, kfold, csv_name)
            #print("\t Prediction tracking done\n", flush=True)

            ## calculate errors and scores
            #errors_and_scores(trainX, trainY, alg1_init, alg2_init, alg3_init, scores, kfold, csv_name)
            #print("\t CV scoring done\n", flush=True)

            ## learning curves
            #learning_curves(trainX, trainY, alg1_init, alg2_init, alg3_init, kfold, csv_name)
            #print("\t Learning curves done\n", flush=True)
            #
            #print("The {} predictions in trainset {} are complete\n".format(parameter, trainset), flush=True)
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", flush=True)

    return

if __name__ == "__main__":
    main()
