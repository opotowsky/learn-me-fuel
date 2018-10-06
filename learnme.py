#! /usr/bin/env python3

from tools import splitXY, top_nucs, filter_nucs

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold, learning_curve, validation_curve

import pandas as pd
import numpy as np

def validation_curves(X, Y, alg1, alg2, alg3, CV, score, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, decision tree, and support vector methods), varying
    the training set size for each prediction category: reactor type, cooling
    time, enrichment, and burnup

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
    CV : cross-validation generator
    score : 
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *validation_curve.csv : csv file with val curve results for each 
                            prediction category

    """    
    
    # Note: I'm trying to avoid loops here so the code is inelegant

    # Varied alg params for validation curves
    k_list = np.linspace(1, 39, 10).astype(int)
    depth_list = np.linspace(10, 100, 10)
    gamma_list = np.logspace(-4, -1, 10)
    c_list = np.linspace(0.1, 100000, 10)

    # knn
    train, cv = validation_curve(alg1, X, Y, 'n_neighbors', k_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df1 = pd.DataFrame({'ParamList' : k_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df1['Algorithm'] = 'knn'

    # dtree
    train, cv = validation_curve(alg2, X, Y, 'max_depth', depth_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df2 = pd.DataFrame({'ParamList' : depth_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df2['Algorithm'] = 'dtree'
    
    # svr
    train, cv = validation_curve(alg3, X, Y, 'gamma', gamma_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df3 = pd.DataFrame({'ParamList' : gamma_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df3['Algorithm'] = 'svr'

    vc_data = pd.concat([df1, df2, df3])
    vc_data.to_csv(csv_name + '_validation_curve.csv')
    return 

def learning_curves(X, Y, alg1, alg2, alg3, CV, score, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, decision tree, and support vector methods), varying
    the training set size for each prediction category: reactor type, cooling
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
    tstd = 'TrainStd'
    cstd = 'CV-Std'

    # knn
    tsize, train, cv = learning_curve(alg1, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df1 = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                        cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    df1['Algorithm'] = 'knn'

    # dtree
    tsize, train, cv = learning_curve(alg2, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df2 = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                        cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    df2['Algorithm'] = 'dtree'
    
    # svr
    tsize, train, cv = learning_curve(alg3, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df3 = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                        cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    df3['Algorithm'] = 'svr'

    lc_data = pd.concat([df1, df2, df3])
    lc_data.index.name = 'TrainSizeFrac'
    lc_data.to_csv(csv_name + '_learning_curve.csv')
    return 


def track_predictions(X, Y, alg1, alg2, alg3, CV, csv_name):
    """
    Saves csv's with predictions of each reactor parameter instance.
    
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
    *predictions.csv : csv file with prediction results 

    """
    knn = cross_val_predict(alg1, X, y=Y, cv=CV, n_jobs=-1)
    dtr = cross_val_predict(alg2, X, y=Y, cv=CV, n_jobs=-1)
    svr = cross_val_predict(alg3, X, y=Y, cv=CV, n_jobs=-1)

    preds_by_alg = pd.DataFrame({'TrueY': Y, 'kNN': knn, 
                                 'DTree': dtr, 'SVR': svr}, 
                                 index=Y.index)
    preds_by_alg.to_csv(csv_name + '_predictions.csv')
    return

def errors_and_scores(X, Y, alg1, alg2, alg3, scores, CV, csv_name):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
    scores : list of scoring types (from sckikit-learn)
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *scores.csv : csv file with scores for each CV fold
    
    """
    
    cv_scr = cross_validate(alg1, X, Y, scoring=scores, cv=CV, 
                            return_train_score=False, n_jobs=-1)
    df1 = pd.DataFrame(cv_scr)
    df1['Algorithm'] = 'knn'
    
    cv_scr = cross_validate(alg2, X, Y, scoring=scores, cv=CV, 
                            return_train_score=False, n_jobs=-1)
    df2 = pd.DataFrame(cv_scr)
    df2['Algorithm'] = 'dtree'
    
    cv_scr = cross_validate(alg3, X, Y, scoring=scores, cv=CV, 
                            return_train_score=False, n_jobs=-1)
    df3 = pd.DataFrame(cv_scr)
    df3['Algorithm'] = 'svr'
    
    cv_results = [df1, df2, df3]
    df = pd.concat(cv_results)
    df.to_csv(csv_name + '_scores.csv')
    
    return

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
    # hyperparam optimization was done on 60% of training set
    trainXY = trainXY.sample(frac=0.6)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    
    # loops through each reactor parameter to do separate predictions
    # burnup is last since its the only tset I'm altering
    for Y in ('r', 'c', 'e', 'b'):
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
        if Y is 'r':
            score = 'accuracy'
            kfold = StratifiedKFold(n_splits=CV, shuffle=True)
            knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
            dtr_init = DecisionTreeClassifier(max_depth=depth, max_features=feats, class_weight='balanced')
            svr_init = SVC(gamma=g, C=c, class_weight='balanced')

        # track predictions 
        track_predictions(trainX, trainY, knn_init, dtr_init, svr_init, kfold, csv_name)

        # calculate errors and scores
        scores = ['r2', 'explained_variance', 'neg_mean_absolute_error']
        if Y is 'r':
            scores = ['accuracy', ]
        errors_and_scores(trainX, trainY, knn_init, dtr_init, svr_init, scores, kfold, csv_name)

        # learning curves
        #learning_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
        
        # validation curves 
        # VC needs different inits
        #score = 'explained_variance'
        #kfold = KFold(n_splits=CV, shuffle=True)
        #knn_init = KNeighborsRegressor(weights='distance')
        #dtr_init = DecisionTreeRegressor()
        #svr_init = SVR(C=c)
        #if Y is 'r':
        #    score = 'accuracy'
        #    kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        #    knn_init = KNeighborsClassifier(weights='distance')
        #    dtr_init = DecisionTreeClassifier(class_weight='balanced')
        #    svr_init = SVC(C=c, class_weight='balanced')
        #validation_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
        
        print("The {} predictions are complete\n".format(parameter), flush=True)

    return

if __name__ == "__main__":
    main()
