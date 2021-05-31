#! /usr/bin/env python3

from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, learning_curve, validation_curve, train_test_split

import pandas as pd
import numpy as np
import sys

njobs = 4

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

def get_sfco_hyperparam(idx, d_or_f):
    """
    This is the most embarassingly hacky thing to do, but it's just for a quick
    test and I don't want to update all the CHTC files

    Using the cv and test_set frac parameters from main() (via script argument)
    as indexing of sorts, since they aren't used at all for the ext_test_set
    func, this func will grab different hyperparameters so that there can be a
    very rough manual adaptation of a validation curve made using real-world
    data.

    Parameters
    ----------
    idx : index of hyperparameters (float b/c this is a bad hack)
    d_or_f : which hyperparam gets varied, where the other is held constant 
             (values of 2 or 3, because this is also a bad hack)

    Returns
    -------
    k : number of nearest neighbors
    depth : depth of decision tree
    feats : number of features in decision tree
    g : gamma for SVM
    c : C for SVM
    
    """
    steps = 10
    k_list = np.linspace(1, 10, steps).astype(int)
    depth_list = np.linspace(25, 85, steps).astype(int)
    #using nuc29 trainset only, max feats is thus 29
    feats_list = np.linspace(9, 29, steps).astype(int)
    gamma_list = [0.01] * steps
    c_list = [10000] * steps
    
    # tfrac-->idx must be 1+, so correcting back to 0 indexing
    i = int(idx) - 1
    k = k_list[i]
    depth = depth_list[i] 
    feats = feats_list[i]
    g = gamma_list[i]
    c = c_list[i]
    
    #cv-->d_or_f needs to be 2+, so magic-coding in 2 v 3 comparison
    if d_or_f == 2:
        # vary d, not f
        feats = None 
    else:
        # vary f, not d
        depth = None
        
    return k, depth, feats, g, c

def get_hyperparam(param, train_name, frac):
    """
    This has gotten messy, but the original intention was to keep track of
    optimization of hyperparameters on training sets of various sizes. The
    nuc_conc opts remain, but it was not feasible to run an optimization on
    every single detector trainset (several energy window lists for each
    detector), so a base one was chosen. If perfecting scikit predictions
    becomes the goal, this will be fixed up.
    
    Parameters
    ----------
    param : reactor parameter being predicted
    train_name : name of training set
    frac : training set fraction to use

    Returns
    -------
    k : number of nearest neighbors
    depth : depth of decision tree
    feats : number of features in decision tree
    g : gamma for SVM
    c : C for SVM

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
    #elif 'd1' in train_name:
    #    hp = {'reactor' :    {'k' : 4, 'depth' : 41, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
    #          'burnup' :     {'k' : 3, 'depth' : 53, 'feats' : 99, 'g' : 0.10, 'c' : 100000},
    #          'cooling' :    {'k' : 3, 'depth' : 31, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
    #          'enrichment' : {'k' : 6, 'depth' : 67, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
    #          }
    #elif 'd2' in train_name:
    #    hp = {'reactor' :    {'k' : 8, 'depth' : 58, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
    #          'burnup' :     {'k' : 4, 'depth' : 70, 'feats' : 106, 'g' : 0.10, 'c' : 100000},
    #          'cooling' :    {'k' : 4, 'depth' : 26, 'feats' : 95, 'g' : 0.10, 'c' : 100000},
    #          'enrichment' : {'k' : 6, 'depth' : 63, 'feats' : 109, 'g' : 0.10, 'c' : 100000},
    #          }
    #elif 'd3' in train_name:
    #    hp = {'reactor' :    {'k' : 6, 'depth' : 53, 'feats' : 26, 'g' : 0.10, 'c' : 100000},
    #          'burnup' :     {'k' : 4, 'depth' : 67, 'feats' : 30, 'g' : 0.10, 'c' : 100000},
    #          'cooling' :    {'k' : 4, 'depth' : 26, 'feats' : 27, 'g' : 0.10, 'c' : 100000},
    #          'enrichment' : {'k' : 7, 'depth' : 38, 'feats' : 24, 'g' : 0.10, 'c' : 100000},
    #          }
    #elif 'd6' in train_name:
    #    hp = {'reactor' :    {'k' : 6, 'depth' : 53, 'feats' : 26, 'g' : 0.10, 'c' : 100000},
    #          'burnup' :     {'k' : 4, 'depth' : 67, 'feats' : 30, 'g' : 0.10, 'c' : 100000},
    #          'cooling' :    {'k' : 4, 'depth' : 26, 'feats' : 27, 'g' : 0.10, 'c' : 100000},
    #          'enrichment' : {'k' : 7, 'depth' : 38, 'feats' : 24, 'g' : 0.10, 'c' : 100000},
    #          }
    else:
        hp = {'reactor' :    {'k' : 5, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              'burnup' :     {'k' : 3, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              'cooling' :    {'k' : 3, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
              'enrichment' : {'k' : 5, 'depth' : None, 'feats' : None, 'g' : 0.10, 'c' : 100000},
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

# TODO this func may be able to be reduced to a single line in the random error func for speediness
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

def random_error(X_unscaled, Y, alg, alg_init, csv_name, param):
    """
    This function has been updated to better mimic the MLL method, by using
    train_test_split for a similar test set fraction. Then, for several values
    of a randomly applied error, the prediction performance is measured. 

    [Old func used cross_validate, but that doesn't allow for apples-to-apples
    comparison against the MLL results.]
    
    Parameters 
    ---------- 
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes
    param : reactor parameter being predicted

    Returns
    -------
    *random_error.csv : csv file with prediction performace with respect to 
                        random error magnitude

    """
    err_percent = [0, 0.1, 0.3, 0.6, 0.9, 
                   1, 1.3, 1.6, 1.9, 2, 
                   2.5, 3, 3.5, 4, 4.5, 5,
                   6, 7, 8, 9, 10, 13, 17, 
                   20]
    prederr = []
    prederr_std = []
    
    for err in err_percent:
        X = add_error(err, X_unscaled)
        X = scale(X)
        # split train and test set to mimic MLL process
        # frac is different than the other func bc MLL sampling was lower for nuc conc
        test_frac = 0.055
        if param == 'reactor': 
            trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_frac, shuffle=True, stratify=Y)
        else:
            trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_frac, shuffle=True)
        alg_init.fit(trainX, trainY)
        preds = alg_init.predict(testX)
        # keeping rand_err output the same by post processing here
        if param == 'reactor':
            errcol = np.where(testY == preds, True, False)
        else:
            errcol = np.abs(testY - preds)
        prederr.append(errcol.mean())
        prederr_std.append(errcol.std())

    if param == 'reactor':
        df = pd.DataFrame({'Percent Error' : err_percent, 
                            algs[alg]+' Acc' : prederr,
                            algs[alg]+' Acc Std' : prederr_std
                            })
    else:
        df = pd.DataFrame({'Percent Error' : err_percent, 
                            algs[alg]+' MAE' : prederr,
                            algs[alg]+' MAE Std' : prederr_std,
                            })
    df.to_csv(csv_name + '_random_error.csv')
    return

def validation_curves(X, Y, alg, alg_init, CV, score, csv_name):
    """
    Given training set, runs an ML algorithm (currently, this is nearest
    neighbor, decision tree, or support vector), varying the hyperparameters of
    the algorithm over a range. 

    Parameters 
    ---------- 
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    CV : cross-validation generator
    score : string of scoring type (from sckikit-learn)
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *validation_curve.csv : csv file with val curve results for each 
                            prediction category

    """    
    steps = 15
    k_list = np.linspace(1, 20, steps).astype(int)
    depth_list = np.linspace(10, 90, steps).astype(int)
    feat_list = np.linspace(10, len(X.columns), steps).astype(int)
    gamma_list = np.logspace(-4, 0, steps)
    c_list = np.logspace(0, 5, steps)

    if alg == 'knn':
        hparam = ['n_neighbors']
        hp_range = [k_list]
    elif alg == 'dtree':
        hparam = ['max_depth', 'max_features']
        hp_range = [depth_list, feat_list]
    else: #svm
        hparam = ['gamma', 'C']
        hp_range = [gamma_list, c_list]

    df_list = []
    for hp, hp_list in zip(hparam, hp_range):
        train, cv = validation_curve(alg_init, X, Y, param_name=hp,
                                     param_range=hp_range, cv=CV, 
                                     scoring=score, n_jobs=njobs)
        train_mean = np.mean(train, axis=1)
        train_std = np.std(train, axis=1)
        cv_mean = np.mean(cv, axis=1)
        cv_std = np.std(cv, axis=1)
        df = pd.DataFrame({'ParamList' : hp_range, 'TrainScore' : train_mean, 
                           'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                           'CV-Std' : cv_std})
        df['Algorithm'] = alg
        df_list.append(df)

    all_vcs = pd.concat(df_list)
    all_vcs.to_csv(csv_name + '_validation_curve.csv')
    return 

def learning_curves(X, Y, alg, alg_init, CV, score, csv_name):
    """
    Given training data, runs an ML algorithm (currently, this is nearest
    neighbor, decision tree, or support vector), varying the training set size
    from 5% to 100%

    Parameters 
    ---------- 
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    CV : cross-validation generator
    score : string of scoring type (from sckikit-learn)
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *learning_curve.csv : csv file with learning curve results for each 
                          prediction category

    """    
    trainset_frac = np.array( [0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 
                               0.5, 0.6, 0.7, 0.8, 0.9, 1.0] )
    col_names = ['AbsTrainSize', 'TrainScore', 'CV-Score']
    tsze = 'AbsTrainSize'
    tscr = 'TrainScore'
    cscr = 'CV-Score'
    tstd = 'TrainStd'
    cstd = 'CV-Std'

    tsize, train, cv = learning_curve(alg_init, X, Y, train_sizes=trainset_frac, 
                                      scoring=score, cv=CV, shuffle=True, 
                                      n_jobs=njobs)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    lc_df = pd.DataFrame({tsze : tsize, tscr : train_mean, tstd : train_std, 
                          cscr : cv_mean, cstd : cv_std}, index=trainset_frac)
    if alg == 'knn':
        lc_df['Algorithm'] = 'knn'
    elif alg == 'dtree':
        lc_df['Algorithm'] = 'dtree'
    else: # svm
        lc_df['Algorithm'] = 'svm'

    lc_df.index.name = 'TrainSizeFrac'
    lc_df.to_csv(csv_name + '_learning_curve.csv')
    return 


def errors_and_scores(X_u, Y, alg, alg_init, score, CV, csv_name, tset_name):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    Parameters 
    ---------- 
    
    X_u : dataframe that includes all training data before scaling
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    score : string of scoring type (from sckikit-learn), or pre-initialized scorer
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being
               predicted for naming purposes
    tset_name : string of the trainset name, to distinguish random error 
                injection. Or in case of PCA, string indicating such

    Returns
    -------
    *scores.csv : csv file with scores for each CV fold
    
    """
    # add "counting" error to summed bins or uniform error to nuc masses
    if 'spectra' in tset_name:
        X = np.random.uniform(X_u - np.sqrt(X_u), X_u + np.sqrt(X_u))
        X = scale(X)
    elif 'pca' in tset_name:
        X = X_u
    else:
        X = add_error(5.0, X_u)
        X = scale(X)
    
    cv_scr = cross_validate(alg_init, X, Y, scoring=score, cv=CV, 
                            return_train_score=False, n_jobs=njobs)
    df = pd.DataFrame(cv_scr)
    df['Algorithm'] = alg
    df.to_csv(csv_name + '_scores.csv')
    
    return

def cv_predict(X_u, Y, alg, alg_init, CV, csv_name, tset_name, pred_param):
    """
    Saves csv's with each prediction from when the sample was in the
    testing CV fold

    Parameters 
    ---------- 
    
    X_u : dataframe that includes all training data before scaling
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being
               predicted for naming purposes
    tset_name : string of the trainset name, to distinguish random error 
                injection. 
    pred_param : reactor parameter being predicted

    Returns
    -------
    *predictions.csv : csv file with predictions from entire train set
    
    """
    # add "counting" error to summed bins or uniform error to nuc masses
    if 'spectra' in tset_name:
        X = np.random.uniform(X_u - np.sqrt(X_u), X_u + np.sqrt(X_u))
    else:
        X = add_error(5.0, X_u)
    X = scale(X)
    
    preds = cross_val_predict(alg_init, X, Y, cv=CV, n_jobs=njobs)
    if pred_param == 'reactor':
        errcol = np.where(Y == preds, True, False)
    else:
        errcol = np.abs(Y - preds)
    df = pd.DataFrame({'TrueY': Y, algs[alg]: preds, 'AbsError': errcol},
                       index=Y.index) 
    df.to_csv(csv_name + '_predictions.csv')
    
    return

def ext_test_compare(X, Y, testX, testY, alg, alg_init, csv_name, pred_param):
    """
    Given training set and an external test set (currently designed for
    sfcompo), tracks the prediction results for a given alg_init.
    
    Parameters 
    ---------- 
    X : dataframe that includes all training data
    Y : series with labels for training data
    testX : dataframe that includes all testing data measurements
    testY : series with labels for testing data (ground truth)
    alg : name of algorithm
    alg_init : initialized learner
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes
    pred_param : reactor parameter being predicted

    Returns
    -------
    *.csv : csv file with each alg's predictions compared to ground truth
    
    """
    alg_init.fit(X, Y)
    preds = alg_init.predict(testX)
    if pred_param == 'reactor':
        errcol = np.where(testY == preds, True, False)
    else:
        errcol = np.abs(testY - preds)
    df = pd.DataFrame({'TrueY': testY, algs[alg]: preds, 'AbsError': errcol},
                       index=testY.index) 
    df.to_csv(csv_name + '_ext_test_compare.csv')
    return

def int_test_compare(X_u, Y, alg, alg_init, csv_name, tset_frac, tset_name, pred_param):
    """
    Saves csv's with predictions of each reactor parameter instance for a test
    set fraction defined in main()
    
    Update: this is designed to mimic MLL, understanding the loss of CV and
    method generalization. 
    
    Parameters 
    ---------- 
    X_u : dataframe that includes all training data (pre-scaled)
    Y : series with labels for training data
    alg : name of algorithm
    alg_init : initialized learner
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes
    tset_frac : TEMP HACKY THING DON'T LOOK AT ME
    tset_name : string of the trainset name, to distinguish random error 
                injection. Or in case of PCA, string indicating such
    pred_param : reactor parameter being predicted

    Returns
    -------
    *predictions.csv : csv file with prediction results 

    """
    # add "counting" error to summed bins or uniform error to nuc masses
    if 'spectra' in tset_name:
        X = np.random.uniform(X_u - np.sqrt(X_u), X_u + np.sqrt(X_u))
    else:
        X = add_error(5.0, X_u)
    X = scale(X)
    # split train and test set to mimic MLL process
    test_frac = tset_frac - 1
    if pred_param == 'reactor': 
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_frac, shuffle=True, stratify=Y)
    else:
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_frac, shuffle=True)
    
    alg_init.fit(trainX, trainY)
    preds = alg_init.predict(testX)
    if pred_param == 'reactor':
        errcol = np.where(testY == preds, True, False)
    else:
        errcol = np.abs(testY - preds)
    df = pd.DataFrame({'TrueY': testY, algs[alg]: preds, 'AbsError': errcol},
                       index=testY.index) 
    df.to_csv(csv_name + '_mimic_mll.csv')
    
    return
