#! /usr/bin/env python3

from tools import splitXY, get_testsetXY, convert_g_to_mgUi, get_hyperparam, get_sfco_hyperparam
from tools import int_test_compare, errors_and_scores, validation_curves, learning_curves, ext_test_compare, random_error, cv_predict

from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd
import argparse
import sys

def parse_args(args):
    """
    Command-line argument parsing

    Parameters
    ----------
    args : 

    Returns
    -------
    args : 

    """

    parser = argparse.ArgumentParser(description='Performs machine learning-based predictions or model selection techniques.')
    parser.add_argument('outfile', metavar='csv-output',  
                        help='name for csv output file')
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
    parser.add_argument('-es', '--err_n_scores', action='store_true', 
                        default=False, help='run the errors_and_scores function')
    parser.add_argument('-lc', '--learn_curves', action='store_true', 
                        default=False, help='run the learning_curves function')
    parser.add_argument('-vc', '--valid_curves', action='store_true', 
                        default=False, help='run the validation_curves function')
    parser.add_argument('-re', '--random_error', action='store_true', 
                        default=False, help='run the random_error function')
    parser.add_argument('-itc', '--int_test_compare', action='store_true', 
                        default=False, help='run the int_test_compare function')
    parser.add_argument('-etc', '--ext_test_compare', action='store_true', 
                        default=False, help='run the ext_test_compare function')
    parser.add_argument('-cvp', '--cv_pred', action='store_true',
                        default=False, help='run the cv_predict function')
    parser.add_argument('-testset', '--testing_set', 
                        help='file path to an external testing set')

    return parser.parse_args(args)

def main():
    """
    Given training data, this script performs a user-defined number of ML 
    tasks for three algorithms (kNN, decision trees, SVR), saving the results
    as .csv files:
    1. cv_predict provides the prediction of each training instance from when
       it was in the CV fold for testing
    2. errors_and_scores provides the prediction accuracy for each algorithm
       for each CV fold
    3. learning_curves provides the prediction accuracy with respect to
       training set size
    4. validation_curves provides the prediction accuracy with respect to
       algorithm hyperparameter variations
    5. ext_test_compare provides the predictions of each algorithm of an 
       external test set
    6. int_test_compare provides the predictions of each algorithm of an 
       from a single split-off-from-DB test set
    7. random_error calculates prediction performance with respect to 
       increasing error

    """
    
    args = parse_args(sys.argv[1:])
    
    CV = args.cv
    alg = args.alg
    tset_frac = args.tset_frac
    csv_name =  args.outfile
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    nonlbls = ['AvgPowerDensity', 'ModDensity', 'UiWeight']
    
    trainXY = pd.read_pickle(args.train_db)
    #trainXY.reset_index(inplace=True, drop=True) 
    if tset_frac < 1.0:
        trainXY = trainXY.sample(frac=tset_frac)
    trainX_unscaled, rY, cY, eY, bY = splitXY(trainXY)
    if ((args.learn_curves == True) or (args.valid_curves == True)):
        trainX = scale(trainX_unscaled)
    
    # set ground truth 
    trainY = pd.Series()
    if args.rxtr_param == 'cooling':
        trainY = cY
    elif args.rxtr_param == 'enrichment': 
        trainY = eY
    elif args.rxtr_param == 'burnup':
        trainY = bY
    else:
        trainY = rY

    # get hyperparams
    #if args.ext_test_compare == True:
    #    # hacking the use of these arguments, sorry future me
    #    k, depth, feats, g, c = get_sfco_hyperparam(tset_frac, CV)
    #else:
    k, depth, feats, g, c = get_hyperparam(args.rxtr_param, args.train_db)
        
    ## initialize learners
    score = 'neg_mean_absolute_error'
    kfold = KFold(n_splits=CV, shuffle=True)
    knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1, metric='minkowski')
    dtr_init = DecisionTreeRegressor(max_depth=depth, max_features=feats)
    svr_init = SVR(gamma=g, C=c)
    if args.rxtr_param == 'reactor':
        score = make_scorer(balanced_accuracy_score, adjusted=True)
        kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1, metric='minkowski')
        dtr_init = DecisionTreeClassifier(max_depth=depth, max_features=feats, class_weight='balanced')
        svr_init = SVC(gamma=g, C=c, class_weight='balanced')
    
    if alg == 'knn':
        init = knn_init
    elif alg == 'dtree':
        init = dtr_init
    else:
        init = svr_init

    ## create test set from train set (no CV)
    if args.int_test_compare == True:
        int_test_compare(trainX_unscaled, trainY, alg, init, csv_name, tset_frac, args.train_db, args.rxtr_param)

    ## calculate errors and scores
    if args.err_n_scores == True:
        errors_and_scores(trainX_unscaled, trainY, alg, init, score, kfold, csv_name, args.train_db)

    ## return all CV predictions
    if args.cv_pred == True:
        cv_predict(trainX_unscaled, trainY, alg, init, kfold, csv_name, args.train_db, args.rxtr_param)

    # learning curves
    if args.learn_curves == True:
        learning_curves(trainX, trainY, alg, init, kfold, score, csv_name)
    
    # compare against external test set
    if args.ext_test_compare == True:
        # convert trainset to be same units as testset
        trainX_unscaled = convert_g_to_mgUi(trainX_unscaled)
        xy_cols = trainXY.columns.tolist()
        for col in nonlbls: xy_cols.remove(col)
        testX_unscaled, testY = get_testsetXY(args.testing_set, xy_cols, args.rxtr_param)
        # scale testset using scale fit from trainset
        scaler = StandardScaler().fit(trainX_unscaled)
        trainX = scaler.transform(trainX_unscaled)
        testX = scaler.transform(testX_unscaled)
        ext_test_compare(trainX, trainY, testX, testY, alg, init, csv_name, args.rxtr_param)

    # pred results wrt random error
    if args.random_error == True:
        random_error(trainX_unscaled, trainY, alg, init, kfold, csv_name, args.rxtr_param)

    # validation curves 
    if args.valid_curves == True:
        # VC needs different inits
        knn_init = KNeighborsRegressor(weights='distance', p=1, metric='minkowski')
        dtr_init = DecisionTreeRegressor()
        svr_init = SVR(gamma='scale', C=c)
        if args.rxtr_param == 'reactor':
            knn_init = KNeighborsClassifier(weights='distance', p=1, metric='minkowski')
            dtr_init = DecisionTreeClassifier(class_weight='balanced')
            svr_init = SVC(gamma='scale', C=c, class_weight='balanced')
        if alg == 'knn':
            init = knn_init
        elif alg == 'dtree':
            init = dtr_init
        else:
            init = svr_init
        validation_curves(trainX, trainY, alg, init, kfold, score, csv_name)
    
    return

if __name__ == "__main__":
    main()
