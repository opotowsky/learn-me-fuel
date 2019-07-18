#!/usr/bin/env python3

from string import Template
from tools import splitXY
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

import pandas as pd
import numpy as np
import argparse

def main():

    CV = 5
    tset_frac = 0.6
    iters = 30
    jobs = 4
    c = 10000
    parser = argparse.ArgumentParser(description='Performs hyperparameter optimization for various machine learning algorithms.')
    parser.add_argument('rxtr_param', choices=['reactor', 'cooling', 'enrichment', 'burnup'], 
                        metavar='prediction-param', help='which reactor parameter is to be predicted [reactor, cooling, enrichment, burnup]')
    parser.add_argument('opt_type', choices=['grid', 'random'], metavar='optimization-type', help='which type of hyperparameter optimization strategy to pursue [grid, random]')
    args = parser.parse_args()
    
    # get data set
    trainset = 'trainset.pkl'
    trainXY = pd.read_pickle(trainset)
    trainXY.reset_index(inplace=True, drop=True) 
    trainXY = trainXY.sample(frac=tset_frac)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    # define search breadth
    knn_grid = {'n_neighbors': np.linspace(1, 40, 20).astype(int)}
    dtr_grid = {"max_depth": np.linspace(3, 100, 20).astype(int),
                "max_features": np.linspace(5, len(trainXY.columns)-6, 20).astype(int)}
    svr_grid = {'C': np.logspace(0, 5, 20), 'gamma': np.logspace(-7, 2, 20)} 
    
    score = 'explained_variance'
    kfold = KFold(n_splits=CV, shuffle=True)
    knn_init = KNeighborsRegressor(weights='distance')
    dtr_init = DecisionTreeRegressor()
    svr_init = SVR(C=c)
    # save results
    param_file = args.rxtr_param + '_' + args.opt_type + '_hyperparameters.txt'
    with open(param_file, 'a') as pf:
        pf.write(args.opt_type + ' hyperparameter optimization for ' + args.rxtr_param + ':')
    
    trainY = pd.Series()
    if args.rxtr_param == 'cooling':
        trainY = cY
        parameter = 'cooling'
    elif args.rxtr_param == 'burnup':
        trainY = bY
        parameter = 'burnup'
    elif args.rxtr_param == 'enrichment': 
        trainY = eY
        parameter = 'enrichment'
    else:
        trainY = rY
        parameter = 'reactor'
        score = 'accuracy'
        kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsClassifier(weights='distance')
        dtr_init = DecisionTreeClassifier(class_weight='balanced')
        svr_init = SVC(C=c, class_weight='balanced')
    
    if args.opt_type == 'grid':
        knn_opt = GridSearchCV(estimator=knn_init, param_grid=knn_grid, 
                               scoring=score, n_jobs=jobs, cv=kfold, 
                               return_train_score=True)
        dtr_opt = GridSearchCV(estimator=dtr_init, param_grid=dtr_grid,
                               scoring=score, n_jobs=jobs, cv=kfold, 
                               return_train_score=True)
        svr_opt = GridSearchCV(estimator=svr_init, param_grid=svr_grid,
                               scoring=score, n_jobs=jobs, cv=kfold, 
                               return_train_score=True)
    else:
        knn_opt = RandomizedSearchCV(estimator=knn_init, param_distributions=knn_grid, 
                                     n_iter=iters, scoring=score, n_jobs=jobs, cv=kfold, 
                                     return_train_score=True)
        dtr_opt = RandomizedSearchCV(estimator=dtr_init, param_distributions=dtr_grid,
                                     n_iter=iters, scoring=score, n_jobs=jobs, cv=kfold, 
                                     return_train_score=True)
        svr_opt = RandomizedSearchCV(estimator=svr_init, param_distributions=svr_grid,
                                     n_iter=iters, scoring=score, n_jobs=jobs, cv=kfold, 
                                     return_train_score=True)
    knn_opt.fit(trainX, trainY)
    dtr_opt.fit(trainX, trainY)
    svr_opt.fit(trainX, trainY)
    
    # save info
    tmpl = Template(
'''
knn
    score: $s1
    n_neighbors: $k
dtree
    score: $s2
    max_depth: $d
    max_features: $f
svr
    score: $s3
    gamma: $g
    C: $c
''')
    txt = tmpl.substitute(s1=knn_opt.best_score_,
                          k=knn_opt.best_params_['n_neighbors'],
                          s2=dtr_opt.best_score_,
                          d=dtr_opt.best_params_['max_depth'],
                          f=dtr_opt.best_params_['max_features'],
                          s3=svr_opt.best_score_,
                          g=svr_opt.best_params_['gamma'],
                          c=svr_opt.best_params_['C'])
    with open(param_file, 'a') as pf:
        pf.write(txt)
    
if __name__ == "__main__":
    main()
