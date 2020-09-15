#! /usr/bin/env python3

from learn.tools import splitXY, get_testsetXY, convert_g_to_mgUi
from learn.tools import track_predictions, errors_and_scores, validation_curves, learning_curves, ext_test_compare, random_error

from sklearn.preprocessing import scale
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
    parser.add_argument('rxtr_param', choices=['reactor', 'cooling', 'enrichment', 'burnup'], 
                        metavar='prediction-param', help='which reactor parameter is to be predicted [reactor, cooling, enrichment, burnup]')
    parser.add_argument('tset_frac', metavar='trainset-fraction', type=float,
                        help='fraction of training set to use in algorithms')
    parser.add_argument('cv', metavar='cv-folds', type=int,
                        help='number of cross validation folds')
    parser.add_argument('train_db', metavar='reactor-db', 
                        help='file path to a training set')
    parser.add_argument('outfile', metavar='csv-output',  
                        help='name for csv output file')
    parser.add_argument('-tp', '--track_preds', action='store_true', 
                        default=False, help='run the track_predictions function')
    parser.add_argument('-es', '--err_n_scores', action='store_true', 
                        default=False, help='run the errors_and_scores function')
    parser.add_argument('-lc', '--learn_curves', action='store_true', 
                        default=False, help='run the learning_curves function')
    parser.add_argument('-vc', '--valid_curves', action='store_true', 
                        default=False, help='run the validation_curves function')
    parser.add_argument('-tc', '--test_compare', action='store_true', 
                        default=False, help='run the ext_test_compare function')
    parser.add_argument('-testset', '--testing_set', 
                        help='file path to an external testing set')
    parser.add_argument('-re', '--random_error', action='store_true', 
                        default=False, help='run the random_error function')

    return parser.parse_args(args)

def main():
    """
    Given training data, this script performs a user-defined number of ML 
    tasks for three algorithms (kNN, decision trees, SVR), saving the results
    as .csv files:
    1. track_predictions provides the prediction of each training instance
    2. errors_and_scores provides the prediction accuracy for each algorithm
    for each CV fold
    3. learning_curves provides the prediction accuracy with respect to
    training set size
    4. validation_curves provides the prediction accuracy with respect to
    algorithm hyperparameter variations
    5. ext_test_compare provides the predictions of each algorithm of an 
    external test set
    6. random_error calculates prediction performance with respect to 
    increasing error

    """
    
    args = parse_args(sys.argv[1:])
    
    CV = args.cv
    tset_frac = args.tset_frac
    csv_name =  args.rxtr_param
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    nonlbls = ['AvgPowerDensity', 'ModDensity', 'UiWeight']
    
    # locally, pkl file location should be: '../../prep-pkls/pkl_dir/file.pkl'
    pkl = args.train_db  
    trainXY = pd.read_pickle(pkl)
    trainXY.reset_index(inplace=True, drop=True) 
    # ensure hyperparam optimization was done on correct tset_frac
    trainXY = trainXY.sample(frac=tset_frac)
    trainX_unscaled, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX_unscaled)
    
    trainY = pd.Series()
    # get param names and set ground truth
    if args.rxtr_param == 'cooling':
        trainY = cY
        k = 3
        depth = 20
        feats = 15
        g = 0.06
        c = 50000
    elif args.rxtr_param == 'enrichment': 
        trainY = eY
        k = 3
        depth = 20
        feats = 15
        g = 0.8
        c = 25000
    elif args.rxtr_param == 'burnup':
        # burnup needs much less training data...this is 24% of data set
        #trainXY = trainXY.sample(frac=0.4)
        #trainX, rY, cY, eY, bY = splitXY(trainXY)
        #trainX = scale(trainX)
        trainY = bY
        k = 3
        depth = 20
        feats = 15
        g = 0.25
        c = 42000
    else:
        trainY = rY
        k = 3
        depth = 20
        feats = 15
        g = 0.07
        c = 1000
        
    ## initialize learners
    score = 'explained_variance'
    kfold = KFold(n_splits=CV, shuffle=True)
    knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
    dtr_init = DecisionTreeRegressor(max_depth=depth, max_features=feats)
    svr_init = SVR(gamma=g, C=c)
    if args.rxtr_param == 'reactor':
        score = 'accuracy'
        kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
        dtr_init = DecisionTreeClassifier(max_depth=depth, max_features=feats, class_weight='balanced')
        svr_init = SVC(gamma=g, C=c, class_weight='balanced')

    ## track predictions
    if args.track_preds == True:
        track_predictions(trainX, trainY, knn_init, dtr_init, svr_init, kfold, csv_name, trainX_unscaled)

    ## calculate errors and scores
    if args.err_n_scores == True:
        scores = ['r2', 'explained_variance', 'neg_mean_absolute_error']
        if args.rxtr_param == 'reactor':
            scores = ['accuracy', ]
        errors_and_scores(trainX, trainY, knn_init, dtr_init, svr_init, scores, kfold, csv_name)

    # learning curves
    if args.learn_curves == True:
        learning_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
    
    # validation curves 
    if args.valid_curves == True:
        # VC needs different inits
        knn_init = KNeighborsRegressor(weights='distance')
        dtr_init = DecisionTreeRegressor()
        svr_init = SVR(gamma='scale', C=c)
        if args.rxtr_param == 'reactor':
            knn_init = KNeighborsClassifier(weights='distance')
            dtr_init = DecisionTreeClassifier(class_weight='balanced')
            svr_init = SVC(gamma='scale', C=c, class_weight='balanced')
        validation_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
    
    # compare against external test set
    if args.test_compare == True:
        trainX_unscaled = convert_g_to_mgUi(trainX_unscaled)
        trainX = scale(trainX_unscaled)
        xy_cols = trainXY.columns.tolist()
        for col in nonlbls+['total']: xy_cols.remove(col)
        testX, testY = get_testsetXY(args.testing_set, xy_cols, args.rxtr_param)
        ext_test_compare(trainX, trainY, testX, testY, knn_init, dtr_init, svr_init, csv_name)

    # pred results wrt random error
    if args.random_error == True:
        random_error(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
    print("The {} predictions are complete\n".format(args.rxtr_param), flush=True)

    return

if __name__ == "__main__":
    main()
