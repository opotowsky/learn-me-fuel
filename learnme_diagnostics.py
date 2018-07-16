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



#def learning_curves(X, rY, cY, eY, bY, nuc_subset):
#    """
#    
#    Given training data, iteratively runs some ML algorithms (currently, this
#    is nearest neighbor, linear, and support vector methods), varying the
#    training set size for each prediction category: reactor type, cooling
#    time, enrichment, and burnup
#
#    Parameters 
#    ---------- 
#    
#    X : dataframe that includes all training data
#    *Y : series with labels for training data
#    nuc_subset : one of all, fission products, actinides, 
#                 or fission products with actinides
#
#    Returns
#    -------
#    learning_reactor*.csv : 
#    learning_cooling*.csv : 
#    learning_enrichment*.csv : 
#    learning_burnup*.csv : 
#
#    """    
#    TRAINSET_SIZES = np.linspace(0.1, 1.0, 19)
#    m_percent = np.linspace(0.15, 1.0, 20)
#    alpha = 0.1
#    rxtr_lc  = lc_classification(X, rY, alpha, cv_folds)
#    cool_lc= lc_regression(X, cY, alpha, cv_folds)
#    enr_lc = lc_regression(X, eY, alpha, cv_folds)
#    burn_lc = lc_regression(X, bY, alpha, cv_folds)
#    r_lc = 'lc_' + 'reactor' + nucs_tracked + train_src + '.csv'
#    c_lc = 'lc_' + 'cooling' + nucs_tracked + train_src + '.csv'
#    e_lc = 'lc_' + 'enrichment' + nucs_tracked + train_src + '.csv'
#    b_lc = 'lc_' + 'burnup' + nucs_tracked + train_src + '.csv'
#    idx_lc = TRAINSET_SIZES
#    pd.DataFrame(rxtr_lc, columns=cols, index=idx_lc).to_csv(r_lc)
#    pd.DataFrame(cool_lc, columns=cols, index=idx_lc).to_csv(c_lc)
#    pd.DataFrame(enr_lc, columns=cols, index=idx_lc).to_csv(e_lc)
#    pd.DataFrame(burn_lc, columns=cols, index=idx_lc).to_csv(b_lc)
#    
#    return 

def validation_curves(X, Y, knn, rr, svr_g, svr_c, scores, kfold, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, linear, and support vector methods), varying the
    algorithm parameters for each prediction category: reactor type, cooling
    time, enrichment, and burnup

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    *Y : series with labels for training data
    scores : list of scoring types (from sckikit-learn)
    kfold : 
    nuc_subset : one of all, fission products, actinides, 
                 or fission products with actinides
    trainset : integer corresponding to training set

    Returns
    -------
    *_validation.csv : csv file with validation curve results for each 
                       prediction category

    """    
    # Varied alg params for validation curves
    k_list = np.linspace(1, 39, 20).astype(int)
    alpha_list = np.logspace(-3, 5, 20)
    gamma_list = np.linspace(0.0005, 0.09, 20)
    c_list = np.linspace(0.1, 100000, 20)

    fname = csv_name + 'validation_curve.csv'
    vc_data = pd.DataFrame()
    col_names = ['ParamList', 'TrainScore', 'CV-Score',
                 'Algorithm', 'Hyperparameter', 'ScoringMetric']
    
    for alg_type in ('knn', 'rr', 'svr_g', 'svr_c'):
        if alg_type == 'knn':
            alg = knn
            param_string = 'n_neighbors'
            param_list = k_list
        elif alg_type == 'rr':
            alg = rr
            param_string = 'alpha'
            param_list = alpha_list
        elif alg_type == 'svr_g':
            alg = svr_g
            param_string = 'gamma'
            param_list = gamma_list
        else:
            alg = svr_c
            param_string = 'C'
            param_list = c_list

        for score in scores:
            train, cv = validation_curve(alg, X, Y, param_string, param_list,
                                         cv=kfold, scoring=score)
            train_mean = np.mean(train, axis=1)
            cv_mean = np.mean(cv, axis=1)
            vc_data['ParamList'] = param_list
            vc_data['TrainScore'] = train_mean
            vc_data['CV-Score'] = cv_mean
            vc_data['Algorithm'] = alg_type
            vc_data['Hyperparameter'] = param_string
            vc_data['ScoringMetric'] = score
            if not os.path.isfile(fname):
                vc_data.to_csv(fname, mode='w', header=col_names)
            else:
                vc_data.to_csv(fname, mode='a', header=False)
    return
    
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
    c = 10000


    pkl_base = './pkl_trainsets/2jul2018/2jul2018_trainset'
    for trainset in ('1', '2'):
        for subset in ('fiss', 'act', 'fissact', 'all'):
            pkl = pkl_base + trainset + '_nucs_' + subset + '_not-scaled.pkl'
            trainXY = pd.read_pickle(pkl, compression=None)
            
            trainX, rY, cY, eY, bY = splitXY(trainXY)
            if subset == 'all':
                top_n = 100
                nuc_set = top_nucs(trainX, top_n)
                trainX = filter_nucs(trainX, nuc_set, top_n)

            # Scale the trainset below. This treatment is for nucs only.
            # For gammas: trainX = scale(trainX, with_mean=False)
            trainX = scale(trainX) 
            
            # loops through each reactor parameter to do separate predictions
            for Y in ('c', 'e', 'b', 'r'):
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
                
                if Y is not 'r':
                    scores = ['r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error']
                    kfold = KFold(n_splits=CV, shuffle=True)
                    knn_init_vc = KNeighborsRegressor(weights='distance')
                    rr_init_vc = Ridge()
                    svr_g_init_vc = SVR(C=c)
                    svr_c_init_vc = SVR(gamma=g)
                    knn_init_lc = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    rr_init_lc = Ridge(alpha=a)
                    svr_init_lc = SVR(gamma=g, C=c)
                else:
                    scores = ['accuracy', ]
                    kfold = StratifiedKFold(n_splits=CV, shuffle=True)
                    knn_init_vc = KNeighborsClassifier(weights='distance')
                    rr_init_vc = RidgeClassifier(class_weight='balanced')
                    svr_g_init_vc = SVC(C=c, class_weight='balanced')
                    svr_c_init_vc = SVC(gamma=g, class_weight='balanced')
                    knn_init_lc = KNeighborsClassifier(n_neighbors=k, weights='distance')
                    rr_init_lc = RidgeClassifier(alpha=a, class_weight='balanced')
                    svr_init_lc = SVC(gamma=g, C=c, class_weight='balanced')
                
                csv_name = 'trainset_' + trainset + '_' + subset + '_' + parameter + '_'
                # Run one diagnostic curve at a time
                validation_curves(trainX, trainY, knn_init_vc, rr_init_vc, 
                                  svr_g_init_vc, svr_c_init_vc, scores, kfold, 
                                  csv_name)
                #learning_curves(trainX, trainY, knn_init_lc, rr_init_lc, 
                #                svr_init_lc, scores, kfold, csv_name)

            print("The {} predictions in trainset {} are complete\n".format(subset, trainset), flush=True)

if __name__ == "__main__":
    main()
