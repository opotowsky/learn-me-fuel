#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs, track_predictions, errors_and_scores, validation_curves, learning_curves, test_set_compare

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd

def main():
    """
    Given training data, this script performs a number of ML tasks for three
    algorithms:
    1. errors_and_scores provides the prediction accuracy for each algorithm
    for each CV fold
    2. train_and_predict provides the prediction of each training instance
    3. validation_curves provides the prediction accuracy with respect to
    algorithm hyperparameter variations
    4. learning_curves  provides the prediction accuracy with respect to
    training set size

    """
    pkl = './pkl_trainsets/2jul2018/22jul2018_trainset3_nucs_fissact_not-scaled.pkl'
    
    # Parameters for the training and predictions
    CV = 5
    
    trainXY = pd.read_pickle(pkl)
    # gotta get rid of duplicate indices
    # this creates an index column...but we don't want this in the training data. deleting for now via drop.
    trainXY.reset_index(inplace=True, drop=True) 
    # hyperparam optimization was done on 60% of training set
    trainXY = trainXY.sample(frac=0.6)
    trainX_unscaled, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX_unscaled)
    
    # loops through each reactor parameter to do separate predictions
    # burnup is last since its the only tset I'm altering
    for Y in ('r', 'b'):# 'b', 'e', 'c'):
        trainY = pd.Series()
        # get param names and set ground truth
        if Y == 'c':
            trainY = cY
            parameter = 'cooling'
            k = 3 #7
            depth = 50 #50, 12
            feats = 25 #36, 47 
            g = 0.06 #0.2
            c = 50000 #200, 75000
        elif Y == 'e': 
            trainY = eY
            parameter = 'enrichment'
            k = 7 #8
            depth = 50 #53, 38
            feats = 25 #33, 16 
            g = 0.8 #0.2
            c = 25000 #420
        elif Y == 'b':
            # burnup needs much less training data...this is 24% of data set
            trainXY = trainXY.sample(frac=0.4)
            trainX, rY, cY, eY, bY = splitXY(trainXY)
            trainX = scale(trainX)
            trainY = bY
            parameter = 'burnup'
            k = 7 #4
            depth = 50 #50, 78
            feats = 25 #23, 42 
            g = 0.25 #0.025
            c = 42000 #105
        else:
            trainY = rY
            parameter = 'reactor'
            k = 3 #1, 2, or 12
            depth = 50 #50, 97
            feats = 25 # 37, 37 
            g = 0.07 #0.2
            c = 1000 #220
        
        csv_name = 'trainset3_fissact_m60_' + parameter
        
        ## initialize learners
        score = 'explained_variance'
        kfold = KFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
        dtr_init = DecisionTreeRegressor(max_depth=depth, max_features=feats)
        svr_init = SVR(gamma=g, C=c)
        if Y == 'r':
            score = 'accuracy'
            kfold = StratifiedKFold(n_splits=CV, shuffle=True)
            knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
            dtr_init = DecisionTreeClassifier(max_depth=depth, max_features=feats, class_weight='balanced')
            svr_init = SVC(gamma=g, C=c, class_weight='balanced')

        ## track predictions 
        track_predictions(trainX, trainY, knn_init, dtr_init, svr_init, kfold, csv_name, trainX_unscaled)

        ## calculate errors and scores
        #scores = ['r2', 'explained_variance', 'neg_mean_absolute_error']
        #if Y is 'r':
        #    scores = ['accuracy', ]
        #errors_and_scores(trainX, trainY, knn_init, dtr_init, svr_init, scores, kfold, csv_name)

        # learning curves
        #learning_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
        
        # validation curves 
        # VC needs different inits
        #score = 'explained_variance'
        #kfold = KFold(n_splits=CV, shuffle=True)
        #knn_init = KNeighborsRegressor(weights='distance')
        #dtr_init = DecisionTreeRegressor()
        #svr_init = SVR()
        #if Y is 'r':
        #    score = 'accuracy'
        #    kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        #    knn_init = KNeighborsClassifier(weights='distance')
        #    dtr_init = DecisionTreeClassifier(class_weight='balanced')
        #    svr_init = SVC(class_weight='balanced')
        #validation_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
       
        # compare against external test set (right now the only one is 
        # Dayman test set)
        ##### 21 Jun 2019: this is untested and may not work without some imports 
        ##### added to tools.py
        #test_set_compare(trainX, trainY, knn_init, dtr_init, svr_init, csv_name)

        #print("The {} predictions are complete\n".format(parameter), flush=True)

    return

if __name__ == "__main__":
    main()
