#! /usr/bin/env python3

from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, learning_curve, validation_curve

import pandas as pd
import numpy as np
import sys

njobs = -1

algs = {'knn' : 'kNN', 'dtree' : 'DTree', 'svm' : 'SVM'}

def splitXY(dfXY):
    """
    Takes a dataframe with all X (features) and Y (labels) information and 
    produces five different pandas datatypes: a dataframe with nuclide info 
    only + a series for each label column.

    Parameters
    ----------
    dfXY : dataframe with nuclide concentraations and 4 labels: reactor type, 
           cooling time, enrichment, and burnup

    Returns
    -------
    dfX : dataframe with only nuclide concentrations for each instance
    rY : dataframe with reactor type for each instance
    cY : dataframe with cooling time for each instance
    eY : dataframe with fuel enrichment for each instance
    bY : dataframe with fuel burnup for each instance
    oY : dataframe with ORIGEN reactor name for each instance

    """

    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    nonlbls = ['AvgPowerDensity', 'ModDensity', 'UiWeight']
    dfX = dfXY.drop(lbls, axis=1)
    for nonlbl in nonlbls+['total']:
        if nonlbl in dfX.columns: 
            dfX.drop(nonlbl, axis=1, inplace=True)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    #o_dfY = dfXY.loc[:, lbls[4]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY

def get_hyperparam(param, train_name, frac):
    """
    
    
    """
    if frac == 1.0:
        # optimized on 100% trainset
        nuc15_hp = {'reactor' :    {'k' : 2, 'depth' : 65, 'feats' : 5, 'g' : 0.07, 'c' : 285000},
                    'burnup' :     {'k' : 2, 'depth' : 34, 'feats' : 14, 'g' : 0.50, 'c' : 100000},
                    'cooling' :    {'k' : 2, 'depth' : 67, 'feats' : 15, 'g' : 0.10, 'c' : 100000},
                    'enrichment' : {'k' : 4, 'depth' : 53, 'feats' : 5, 'g' : 0.10, 'c' : 100000},
                    }
        nuc29_hp = {'reactor' :    {'k' : 2, 'depth' : 73, 'feats' : 11, 'g' : 0.07, 'c' : 23000},
                    'burnup' :     {'k' : 2, 'depth' : 77, 'feats' : 27, 'g' : 0.50, 'c' : 40000},
                    'cooling' :    {'k' : 2, 'depth' : 48, 'feats' : 29, 'g' : 0.01, 'c' : 40000},
                    'enrichment' : {'k' : 2, 'depth' : 77, 'feats' : 24, 'g' : 0.00005, 'c' : 40000},
                    }
    elif frac == 0.5:
        # optimized on 50% trainset
        nuc15_hp = {'reactor' :    {'k' : 2, 'depth' : 50, 'feats' : 6, 'g' : 0.07, 'c' : 285000},
                    'burnup' :     {'k' : 2, 'depth' : 43, 'feats' : 12, 'g' : 0.50, 'c' : 100000},
                    'cooling' :    {'k' : 2, 'depth' : 46, 'feats' : 15, 'g' : 0.10, 'c' : 100000},
                    'enrichment' : {'k' : 4, 'depth' : 31, 'feats' : 5, 'g' : 0.10, 'c' : 100000},
                    }
        nuc29_hp = {'reactor' :    {'k' : 2, 'depth' : 60, 'feats' : 9, 'g' : 0.07, 'c' : 23000},
                    'burnup' :     {'k' : 2, 'depth' : 34, 'feats' : 26, 'g' : 0.50, 'c' : 40000},
                    'cooling' :    {'k' : 2, 'depth' : 75, 'feats' : 27, 'g' : 0.01, 'c' : 40000},
                    'enrichment' : {'k' : 2, 'depth' : 77, 'feats' : 24, 'g' : 0.00005, 'c' : 40000},
                    }
    else:
        # optimized on 10% trainset
        nuc15_hp = {'reactor' :    {'k' : 2, 'depth' : 33, 'feats' : 10, 'g' : 0.07, 'c' : 285000},
                    'burnup' :     {'k' : 9, 'depth' : 54, 'feats' : 14, 'g' : 0.50, 'c' : 100000},
                    'cooling' :    {'k' : 3, 'depth' : 56, 'feats' : 14, 'g' : 0.10, 'c' : 100000},
                    'enrichment' : {'k' : 2, 'depth' : 42, 'feats' : 9, 'g' : 0.10, 'c' : 100000},
                    }
        nuc29_hp = {'reactor' :    {'k' : 3, 'depth' : 70, 'feats' : 17, 'g' : 0.07, 'c' : 23000},
                    'burnup' :     {'k' : 6, 'depth' : 42, 'feats' : 29, 'g' : 0.50, 'c' : 40000},
                    'cooling' :    {'k' : 5, 'depth' : 48, 'feats' : 29, 'g' : 0.01, 'c' : 40000},
                    'enrichment' : {'k' : 3, 'depth' : 71, 'feats' : 25, 'g' : 0.00005, 'c' : 40000},
                    }
    # multiple opt runs on diff sized trainsets for nuc conc trainsets
    if '15' in train_name:
        hp = nuc15_hp
    elif '29' in train_name:
        hp = nuc29_hp
    # processed gamma spec are optimized on 20% trainset. no SVM opt done.
    elif 'd1' in train_name:
        hp = {'reactor' :    {'k' : 4, 'depth' : 41, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              'burnup' :     {'k' : 3, 'depth' : 53, 'feats' : 99, 'g' : 0.10, 'c' : 100000},
              'cooling' :    {'k' : 3, 'depth' : 31, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
              'enrichment' : {'k' : 6, 'depth' : 67, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              }
    elif 'd2' in train_name:
        hp = {'reactor' :    {'k' : 4, 'depth' : 41, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              'burnup' :     {'k' : 3, 'depth' : 53, 'feats' : 100, 'g' : 0.10, 'c' : 100000},
              'cooling' :    {'k' : 3, 'depth' : 31, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
              'enrichment' : {'k' : 6, 'depth' : 67, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              }
    elif 'd3' in train_name:
        hp = {'reactor' :    {'k' : 4, 'depth' : 41, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              'burnup' :     {'k' : 3, 'depth' : 53, 'feats' : 100, 'g' : 0.10, 'c' : 100000},
              'cooling' :    {'k' : 3, 'depth' : 31, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
              'enrichment' : {'k' : 6, 'depth' : 67, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              }
    elif 'd6' in train_name:
        hp = {'reactor' :    {'k' : 4, 'depth' : 41, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              'burnup' :     {'k' : 3, 'depth' : 53, 'feats' : 100, 'g' : 0.10, 'c' : 100000},
              'cooling' :    {'k' : 3, 'depth' : 31, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
              'enrichment' : {'k' : 6, 'depth' : 67, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
              }
    else:
        # or should I sys exit?
        hp = {'reactor' :    {'k' : 2, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              'burnup' :     {'k' : 2, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              'cooling' :    {'k' : 2, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              'enrichment' : {'k' : 2, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              }

    k = hp[param]['k']
    depth = hp[param]['depth']
    feats = hp[param]['feats']
    g = hp[param]['g']
    c = hp[param]['c']

    return k, depth, feats, g, c

def convert_g_to_mgUi(X):
    """
    Converts nuclides from ORIGEN simulations measured in grams to 
    concentrations measured in mg / gUi

    Parameters
    ----------
    X : dataframe of origen sims with nuclides measured in grams

    Returns
    -------
    X : dataframe of origen sims with nuclides measured in mg / gUi
    
    """
    
    nucs = X.columns.tolist()
    # [x (g) / 1e6 (gUi)] * [1000 (mg) / 1 (g)] = x / 1000
    X[nucs] = X[nucs].div(1000, axis=0)
    return X

def get_testsetXY(pklfile, xy_cols, rxtr_param):
    """
    
    """
    
    testXY = pd.read_pickle(pklfile)
    
    # In-script test: order of columns must match:
    if xy_cols != testXY.columns.tolist():
        if sorted(xy_cols) == sorted(testXY.columns.tolist()):
            testXY = testXY[xy_cols]
        else:
            sys.exit('Feature sets are different')
    
    testXY.reset_index(inplace=True, drop=True) 
    testX, rY, cY, eY, bY = splitXY(testXY)
    testY = pd.Series()
    if rxtr_param == 'cooling':
        testY = cY
    elif rxtr_param == 'enrichment':
        testY = eY
    elif rxtr_param == 'burnup':
        testY = bY
    else:
        testY = rY

    return testX, testY

def top_nucs(df, top_n):
    """
    loops through the rows of a dataframe and keeps a list of the top_n 
    nuclides (by concentration) from each row
    
    Parameters
    ----------
    df : dataframe of nuclide concentrations
    top_n : number of nuclides to sort and filter by

    Returns
    -------
    nuc_set : set of the top_n nucs as determined 

    """
    
    # Get a set of top n nucs from each row (instance)
    nuc_set = set()
    for case, conc in df.iterrows():
        top_n_series = conc.sort_values(ascending=False)[:top_n]
        nuc_list = list(top_n_series.index.values)
        nuc_set.update(nuc_list)
    return nuc_set

def filter_nucs(df, nuc_set, top_n):
    """
    for each instance (row), keep only top n values, replace rest with 0
    
    Parameters
    ----------
    df : dataframe of nuclide concentrations
    nuc_set : set of top_n nuclides
    top_n : number of nuclides to sort and filter by

    Returns
    -------
    top_n_df : dataframe that has values only for the top_n nuclides of the set 
               nuc_set in each row

    """
    
    # To filter further, have to reconstruct the df into a new one
    # Found success appending each row to a new df as a series
    top_n_df = pd.DataFrame(columns=tuple(nuc_set))
    for case, conc in df.iterrows():
        top_n_series = conc.sort_values(ascending=False)[:top_n]
        nucs = top_n_series.index.values
        # some top values in test set aren't in nuc set, so need to delete those
        del_list = list(set(nucs) - nuc_set)
        top_n_series.drop(del_list, inplace=True)
        filtered_row = conc.filter(items=top_n_series.index.values)
        top_n_df = top_n_df.append(filtered_row)
    # replace NaNs with 0, bc scikit don't take no NaN
    top_n_df.fillna(value=0, inplace=True)
    return top_n_df

def add_error(percent_err, df):
    """
    Given a dataframe of nuclide vectors, add error to each element in each
    nuclide vector that has a random value within the range [1-err, 1+err]

    Parameters
    ----------
    percent_err : a float indicating the maximum error that can be added to the nuclide
                  vectors
    df : dataframe of only nuclide concentrations

    Returns
    -------
    df_err : dataframe with nuclide concentrations altered by some error

    """
    x = df.shape[0]
    y = df.shape[1]
    err = percent_err / 100.0
    low = 1 - err
    high = 1 + err
    errs = np.random.uniform(low, high, (x, y))
    df_err = df * errs

    return df_err

def random_error(X_unscaled, Y, alg, alg_init, CV, scores, csv_name, param):
    """
    """
    err_percent = [0, 0.1, 0.3, 0.6, 0.9, 
                   1, 1.3, 1.6, 1.9, 2, 
                   2.5, 3, 3.5, 4, 4.5, 5,
                   6, 7, 8, 9, 10, 13, 17, 
                   20]
    acc = []
    acc_std = []
    exv = []
    exv_std = []
    mae = []
    mae_std = []
    rms = []
    rms_std = []
    
    acc_name = 'test_score'
    exv_name = 'test_' + scores[0]
    mae_name = 'test_' + scores[1]
    rms_name = 'test_' + scores[2]
    for err in err_percent:
        X = add_error(err, X_unscaled)
        X = scale(X)
        if alg == 'knn':
            if param == 'reactor':
                knn_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, n_jobs=njobs)
                acc.append(knn_scr[acc_name].mean())
                acc_std.append(knn_scr[acc_name].std())
            else:
                knn_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, n_jobs=njobs)
                exv.append(knn_scr[exv_name].mean())
                exv_std.append(knn_scr[exv_name].std())
                mae.append(knn_scr[mae_name].mean())
                mae_std.append(knn_scr[mae_name].std())
                rms.append(knn_scr[rms_name].mean())
                rms_std.append(knn_scr[rms_name].std())
        else:
            if param == 'reactor':
                dtr_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, n_jobs=njobs)
                acc.append(dtr_scr[acc_name].mean())
                acc_std.append(dtr_scr[acc_name].std())
            else:
                dtr_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, n_jobs=njobs)
                exv.append(dtr_scr[exv_name].mean())
                exv_std.append(dtr_scr[exv_name].std())
                mae.append(dtr_scr[mae_name].mean())
                mae_std.append(dtr_scr[mae_name].std())
                rms.append(dtr_scr[rms_name].mean())
                rms_std.append(dtr_scr[rms_name].std())
    
    if param == 'reactor':
        df = pd.DataFrame({'Percent Error' : err_percent, 
                            algs[alg]+' Acc' : acc,
                            algs[alg]+' Acc Std' : acc_std
                            })
    else:
        df = pd.DataFrame({'Percent Error' : err_percent, 
                            algs[alg]+' ExpVar' : exv,
                            algs[alg]+' ExpVar Std' : exv_std,
                            algs[alg]+' MAE' : mae,
                            algs[alg]+' MAE Std' : mae_std,
                            algs[alg]+' RMSE' : rms,
                            algs[alg]+' RMSE Std' : rms_std
                            })
    df.to_csv(csv_name + '_random_error.csv')
    return

# TODO fix the datframe based on single score --> scores
def validation_curves(X, Y, alg, alg_init, CV, scores, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, decision tree, and support vector methods), varying
    the training set size for each prediction category: reactor type, cooling
    time, enrichment, and burnup

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    CV : cross-validation generator
    scores : 
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *validation_curve.csv : csv file with val curve results for each 
                            prediction category

    """    
    
    # TODO settle on lists!
    steps = 15
    k_list = np.linspace(1, 30, steps).astype(int)
    depth_list = np.linspace(3, 30, steps).astype(int)
    feat_list = np.linspace(1, 15, steps).astype(int)
    gamma_list = np.logspace(-4, 0, steps)
    c_list = np.logspace(0, 5, steps)

    if alg == 'knn':
        train, cv = validation_curve(alg_init, X, Y, param_name='n_neighbors', 
                                     param_range=k_list, cv=CV, 
                                     scoring=scores, n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        df1 = pd.DataFrame({'ParamList' : k_list, 'TrainScore' : train_mean, 
                            'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                            'CV-Std' : cv_std})
        df1['Algorithm'] = 'knn'

        vc_data = df1

    elif alg == 'dtree':
        train, cv = validation_curve(alg_init, X, Y, param_name='max_depth', 
                                     param_range=depth_list, cv=CV, 
                                     scoring=scores, n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        df2 = pd.DataFrame({'ParamList' : depth_list, 'TrainScore' : train_mean, 
                            'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                            'CV-Std' : cv_std})
        df2['Algorithm'] = 'dtree'
        
        train, cv = validation_curve(alg_init, X, Y, param_name='max_features', 
                                     param_range=feat_list, cv=CV, 
                                     scoring=scores, n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        df3 = pd.DataFrame({'ParamList' : feat_list, 'TrainScore' : train_mean, 
                            'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                            'CV-Std' : cv_std})
        df3['Algorithm'] = 'dtree'
        
        vc_data = pd.concat([df2, df3])

    else: # svm
        train, cv = validation_curve(alg_init, X, Y, param_name='gamma', 
                                     param_range=gamma_list, cv=CV, 
                                     scoring=scores, n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        df4 = pd.DataFrame({'ParamList' : gamma_list, 'TrainScore' : train_mean, 
                            'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                            'CV-Std' : cv_std})
        df4['Algorithm'] = 'svm'

        train, cv = validation_curve(alg_init, X, Y, param_name='C', 
                                     param_range=c_list, cv=CV, 
                                     scoring=scores, n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        df5 = pd.DataFrame({'ParamList' : c_list, 'TrainScore' : train_mean, 
                            'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                            'CV-Std' : cv_std})
        df5['Algorithm'] = 'svm'
        
        vc_data = pd.concat([df4, df5])

    vc_data.to_csv(csv_name + '_validation_curve.csv')
    return 

# TODO fix the datframe based on single score --> scores
def learning_curves(X, Y, alg, alg_init, CV, scores, csv_name):
    """
    
    Given training data, iteratively runs some ML algorithms (currently, this
    is nearest neighbor, decision tree, and support vector methods), varying
    the training set size for each prediction category: reactor type, cooling
    time, enrichment, and burnup

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *learning_curve.csv : csv file with learning curve results for each 
                          prediction category

    """    
    
    trainset_frac = np.array( [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 
                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] )
    col_names = ['AbsTrainSize', 'TrainScore', 'CV-Score']
    tsze = 'AbsTrainSize'
    tscr = 'TrainScore'
    cscr = 'CV-Score'
    tstd = 'TrainStd'
    cstd = 'CV-Std'

    if alg == 'knn':
        tsize, train, cv = learning_curve(alg_init, X, Y, train_sizes=trainset_frac, 
                                          scoring=scores, cv=CV, shuffle=True, 
                                          n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        lc_df = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                              cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
        lc_df['Algorithm'] = 'knn'
    elif alg == 'dtree':
        tsize, train, cv = learning_curve(alg_init, X, Y, train_sizes=trainset_frac, 
                                          scoring=scores, cv=CV, shuffle=True, 
                                          n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        lc_df = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                              cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
        lc_df['Algorithm'] = 'dtree'
    else: # svm
        tsize, train, cv = learning_curve(alg_init, X, Y, train_sizes=trainset_frac, 
                                          scoring=scores, cv=CV, shuffle=True, 
                                          n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        lc_df = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                              cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
        lc_df['Algorithm'] = 'svm'

    lc_df.index.name = 'TrainSizeFrac'
    lc_df.to_csv(csv_name + '_learning_curve.csv')
    return 


def track_predictions(X, Y, alg, alg_init, CV, csv_name, X_cols):
    """
    Saves csv's with predictions of each reactor parameter instance.
    
    Parameters 
    ---------- 
    
    X : numpy array that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes
    X_cols : list of nuclide columns

    Returns
    -------
    *predictions.csv : csv file with prediction results 

    """
    X = pd.DataFrame(X, index=Y.index, columns=X_cols)
    if alg == 'knn':
        knn = cross_val_predict(alg_init, X, y=Y, cv=CV, n_jobs=njobs)
        X[algs[alg]] = knn 
    elif alg == 'dtree':
        dtr = cross_val_predict(alg_init, X, y=Y, cv=CV, n_jobs=njobs)
        X[algs[alg]] = dtr 
    else: # svm
        svr = cross_val_predict(alg_init, X, y=Y, cv=CV, n_jobs=njobs)
        X[algs[alg]] = svr
    X.to_csv(csv_name + '_predictions.csv')
    return

def errors_and_scores(X, Y, alg, alg_init, scores, CV, csv_name):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    Parameters 
    ---------- 
    
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    scores : list of scoring types (from sckikit-learn)
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *scores.csv : csv file with scores for each CV fold
    
    """
    
    if alg == 'knn':
        cv_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, 
                                return_train_score=False, n_jobs=njobs)
        df = pd.DataFrame(cv_scr)
        df['Algorithm'] = 'knn'
    elif alg == 'dtree':
        cv_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, 
                                return_train_score=False, n_jobs=njobs)
        df = pd.DataFrame(cv_scr)
        df['Algorithm'] = 'dtree'
    else: # svm
        cv_scr = cross_validate(alg_init, X, Y, scoring=scores, cv=CV, 
                                return_train_score=False, n_jobs=njobs)
        df = pd.DataFrame(cv_scr)
        df['Algorithm'] = 'svm'
    
    df.to_csv(csv_name + '_scores.csv')
    
    return

def ext_test_compare(X, Y, testX, testY, alg, alg_init, csv_name):
    """
    X : dataframe that includes all training data
    Y : series with labels for training data
    testX : dataframe that includes all testing data measurements
    testY : series with labels for testing data (ground truth)
    alg : name of algorithm
    alg_init : initialized learner
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *.csv : csv file with each alg's predictions compared to ground truth
    
    """
    alg_init.fit(X, Y)
    preds = alg_init.predict(testX)
    alg_preds = pd.DataFrame({'TrueY': testY, algs[alg]: preds}, 
                              index=testY.index)
    alg_preds.to_csv(csv_name + '_ext_test_compare.csv')
    return
