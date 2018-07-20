#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import learning_curve, validation_curve, KFold, StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn import metrics

from math import sqrt
import numpy as np
import pandas as pd
import os

def main():
    """
    Takes the pickle file of the training set, splits the dataframe into the 
    appropriate X and Ys for prediction of reactor type, cooling time, fuel 
    enrichment, and burnup. Scales the training data set for the algorithms, then 
    calls for the training and prediction. Saves all results as .csv files. 
            # Generally, you want to peek at validation curve results to set 
            # the best params for the learning curves, perhaps even
            # with one more iteration (i.e. if trainset is too big, give a 
            # fraction of it to the validation curve). Thus, only run this script 
            # one at time
    """
    # Parameters for all trainings
    CV = 10
    # Constant alg params for learning curves
    k = 13
    a = 100
    g = 0.001
    c = 1000
    gamma_list = np.linspace(0.0005, 0.09, 20)
    c_list = np.linspace(0.1, 100000, 20)


    pkl_base = './pkl_trainsets/2jul2018/2jul2018_trainset'
    #for trainset in ('1', '2'):
        #for subset in ('fiss', 'act', 'fissact', 'all'):
    trainset = '2'
    subset = 'fissact'
    pkl = pkl_base + trainset + '_nucs_' + subset + '_not-scaled.pkl'
    trainXY = pd.read_pickle(pkl, compression=None)
    
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    if subset == 'all':
        top_n = 50
        nuc_set = top_nucs(trainX, top_n)
        trainX = filter_nucs(trainX, nuc_set, top_n)

    # Scale the trainset below. This treatment is for nucs only.
    # For gammas: trainX = scale(trainX, with_mean=False)
    trainX = scale(trainX) 
    
    # loops through each reactor parameter to do separate predictions
    for Y in ('b', 'c', 'e'): #'r',
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
        
        #scores = ['r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        score = 'neg_mean_absolute_error'
        kfold = KFold(n_splits=CV, shuffle=True)
        alg = SVR(C=c)
        if Y is 'r':
            score = 'accuracy'
            kfold = StratifiedKFold(n_splits=CV, shuffle=True)
            alg = SVC(C=c, class_weight='balanced')
        
        csv_name = 'trainset_' + trainset + '_' + subset + '_' + parameter + '_'
        fname = csv_name + 'validation_curve.csv'
        vc_data = pd.DataFrame()
        col_names = ['ParamList', 'TrainScore', 'CV-Score',
                     'Algorithm', 'Hyperparameter', 'ScoringMetric']
        param_string = 'gamma'
        param_list = gamma_list
        #param_string = 'C'
        #param_list = c_list
        train, cv = validation_curve(alg, trainX, trainY, param_string, param_list,
                                     cv=kfold, scoring=score)
        train_mean = np.mean(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        vc_data['ParamList'] = param_list
        vc_data['TrainScore'] = train_mean
        vc_data['CV-Score'] = cv_mean
        vc_data['Algorithm'] = 'svr'
        vc_data['Hyperparameter'] = param_string
        vc_data['ScoringMetric'] = score
        if not os.path.isfile(fname):
            vc_data.to_csv(fname, mode='w', header=col_names)
        else:
            vc_data.to_csv(fname, mode='a', header=False)

    print("The {} predictions in trainset {} are complete\n".format(subset, trainset), flush=True)

if __name__ == "__main__":
    main()
