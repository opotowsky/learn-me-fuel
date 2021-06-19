#!/usr/bin/env python3

from tools import splitXY

from string import Template
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

import pandas as pd
import numpy as np
import argparse
import sys

def opt_knn(grid, trainX, trainY, arg):
    """
    Optimizes the k nearest neighbor algorithm, prints the top score and 
    hyperparameter in a text file.

    Parameters
    ----------
    grid : a dict of hyperparameters and range of them to optimize over
    trainX : feature set of nuclide measurements
    trainY : parameter being predicted (e.g., burnup, cooling time)
    arg : a dict of various parameters needed for GridSearchCV and output file

    """
    knn_init = KNeighborsRegressor(weights='distance', p=1, metric='minkowski')
    if arg['pred'] == 'reactor': 
        knn_init = KNeighborsClassifier(weights='distance', p=1, metric='minkowski')
    knn_opt = GridSearchCV(estimator=knn_init, param_grid=grid, 
                           scoring=arg['score'], n_jobs=arg['jobs'], 
                           cv=arg['kfold'], return_train_score=True)
    knn_opt.fit(trainX, trainY) 
    
    tmpl = Template(
'''
score: $score
n_neighbors: $k
''')
    txt = tmpl.substitute(score = knn_opt.best_score_,
                          k = knn_opt.best_params_['n_neighbors'])                  
    with open(arg['file'], 'a') as f:
        f.write(txt)
    return

def opt_dtr(grid, trainX, trainY, arg):
    """
    Optimizes the decision tree algorithm, prints the top score and 
    hyperparameters in a text file.

    Parameters
    ----------
    grid : a dict of hyperparameters and range of them to optimize over
    trainX : feature set of nuclide measurements
    trainY : parameter being predicted (e.g., burnup, cooling time)
    arg : a dict of various parameters needed for GridSearchCV and output file

    """
    dtr_init = DecisionTreeRegressor()
    if arg['pred'] == 'reactor': 
        dtr_init = DecisionTreeClassifier(class_weight='balanced')
    dtr_opt = GridSearchCV(estimator=dtr_init, param_grid=grid,
                           scoring=arg['score'], n_jobs=arg['jobs'], 
                           cv=arg['kfold'], return_train_score=True)
    dtr_opt.fit(trainX, trainY)

    tmpl = Template(
'''
score: $score
max_depth: $d
max_features: $f
''')
    txt = tmpl.substitute(score = dtr_opt.best_score_,
                          d = dtr_opt.best_params_['max_depth'],
                          f = dtr_opt.best_params_['max_features'])
    with open(arg['file'], 'a') as f:
        f.write(txt)
    return

def opt_svm(grid, trainX, trainY, arg):
    """
    Optimizes the support vector machine algorithm, prints the top score and
    hyperparameters in a text file.

    Parameters
    ----------
    grid : a dict of hyperparameters and range of them to optimize over
    trainX : feature set of nuclide measurements
    trainY : parameter being predicted (e.g., burnup, cooling time)
    arg : a dict of various parameters needed for GridSearchCV and output file

    """
    svr_init = SVR(C=arg['c'])
    if arg['pred'] == 'reactor': 
        svr_init = SVC(C=arg['c'], class_weight='balanced')
    svr_opt = GridSearchCV(estimator=svr_init, param_grid=grid,
                           scoring=arg['score'], n_jobs=arg['jobs'], 
                           cv=arg['kfold'], return_train_score=True)
    svr_opt.fit(trainX, trainY)
    tmpl = Template(
'''
score: $score
gamma: $g
C: $c
''')
    txt = tmpl.substitute(score = svr_opt.best_score_,
                          g = svr_opt.best_params_['gamma'],
                          c = svr_opt.best_params_['C'])
    with open(arg['file'], 'a') as f:
        f.write(txt)
    return

def parse_args(args):
    """
    Command-line argument parsing

    Parameters
    ----------
    args : system arguments entered on command line

    Returns
    -------
    args : parsed arguments

    """

    parser = argparse.ArgumentParser(description='Performs hyperparameter optimization for various machine learning algorithms.')
    
    parser.add_argument('rxtr_param', choices=['reactor', 'cooling', 'enrichment', 'burnup'], 
                        metavar='prediction-param', 
                        help='which reactor parameter is to be predicted [reactor, cooling, enrichment, burnup]')
    parser.add_argument('alg', choices=['knn', 'dtree', 'svm'], 
                        metavar='aglortithm', 
                        help='which algorithm to optimize [knn, dtree, svm]')
    parser.add_argument('tset_frac', metavar='trainset-fraction', type=float,
                        help='fraction of training set to use in algorithms')
    parser.add_argument('cv', metavar='cv-folds', type=int,
                        help='number of cross validation folds')
    parser.add_argument('train_db', metavar='reactor-db', 
                        help='file path to a training set')

    return parser.parse_args(args)


def main():
    """
    For a training set of nuclide measurements (features) and reactor
    parameters (labels), this script optimizes the hyperparameters of three
    chosen algorithms given a training set fraction and number of
    cross-validation folds. The results are printed to text files. 
    
    """

    args = parse_args(sys.argv[1:])
    
    CV = args.cv
    tset_frac = args.tset_frac
    
    iters = 20
    jobs = 4
    c = 50000
    
    # get data set
    trainset = args.train_db
    trainXY = pd.read_pickle(trainset)
    trainXY.reset_index(inplace=True, drop=True) 
    trainXY = trainXY.sample(frac=tset_frac)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    
    # define search breadth
    knn_grid = {'n_neighbors': np.linspace(1, 41, iters).astype(int)}
    dtr_grid = {"max_depth": np.linspace(5, 80, iters).astype(int),
                "max_features": np.linspace(5, len(trainXY.columns)-8, iters).astype(int)}
    svr_grid = {'C': np.logspace(0, 6, iters), 'gamma': np.logspace(-7, 1, iters)} 
    
    score = 'neg_mean_absolute_error'
    kfold = KFold(n_splits=CV, shuffle=True)
    trainY = pd.Series()
    if args.rxtr_param == 'cooling':
        trainY = cY
    elif args.rxtr_param == 'burnup':
        trainY = bY
    elif args.rxtr_param == 'enrichment': 
        trainY = eY
    else:
        trainY = rY
        score = 'accuracy'
        kfold = StratifiedKFold(n_splits=CV, shuffle=True)
    
    # save results
    outfile = args.rxtr_param + '_' + args.alg + '_tfrac' + str(args.tset_frac) + '_hyperparameters.txt'
    with open(outfile, 'a') as of:
        of.write(args.alg + ' hyperparameter optimization for ' + args.rxtr_param + ':')
    
    arg_dict = {'score' : score, 'jobs' : jobs, 'kfold' : kfold,
                'pred' : args.rxtr_param, 'file' : outfile, 'c' : c}

    if args.alg == 'knn':
        opt_knn(knn_grid, trainX, trainY, arg_dict)
    elif args.alg == 'dtree':
        opt_dtr(dtr_grid, trainX, trainY, arg_dict)
    else:
        opt_svm(svr_grid, trainX, trainY, arg_dict)
    
    return

if __name__ == "__main__":
    main()
