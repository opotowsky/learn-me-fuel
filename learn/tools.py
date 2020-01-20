#! /usr/bin/env python3

from sklearn.model_selection import cross_val_predict, cross_validate, learning_curve, validation_curve

import pandas as pd
import numpy as np

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
    dfX = dfXY.drop(lbls, axis=1)
    if 'total' in dfX.columns:
        dfX.drop('total', axis=1, inplace=True)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    #o_dfY = dfXY.loc[:, lbls[4]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY

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
    k_list = np.linspace(1, 20, 15).astype(int)
    depth_list = np.linspace(1, 15, 15).astype(int)
    feat_list = np.linspace(1, 15, 15).astype(int)
    gamma_list = np.logspace(-4, -1, 15)
    c_list = np.logspace(0, 5, 15)

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
    
    train, cv = validation_curve(alg2, X, Y, 'max_features', feat_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df3 = pd.DataFrame({'ParamList' : feat_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df3['Algorithm'] = 'dtree'
    
    # svr
    train, cv = validation_curve(alg3, X, Y, 'gamma', gamma_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df4 = pd.DataFrame({'ParamList' : gamma_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df4['Algorithm'] = 'svr'

    train, cv = validation_curve(alg3, X, Y, 'C', c_list, cv=CV, 
                                 scoring=score, n_jobs=-1)
    train_mean = np.mean(train, axis=1)
    train_std = np.std(train, axis=1)
    cv_mean = np.mean(cv, axis=1)
    cv_std = np.std(cv, axis=1)
    df5 = pd.DataFrame({'ParamList' : c_list, 'TrainScore' : train_mean, 
                        'TrainStd' : train_std, 'CV-Score' : cv_mean, 
                        'CV-Std' : cv_std})
    df5['Algorithm'] = 'svr'

    vc_data = pd.concat([df1, df2, df3, df4, df5])
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


def track_predictions(X, Y, alg1, alg2, alg3, CV, csv_name, X_unscaled):
    """
    Saves csv's with predictions of each reactor parameter instance.
    
    Parameters 
    ---------- 
    
    X : numpy array that includes all training data
    Y : series with labels for training data
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
    CV : cross-validation generator
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes
    X_unscaled : dataframe with unscaled nuclide concentrations

    Returns
    -------
    *predictions.csv : csv file with prediction results 

    """
    knn = cross_val_predict(alg1, X, y=Y, cv=CV, n_jobs=-1)
    dtr = cross_val_predict(alg2, X, y=Y, cv=CV, n_jobs=-1)
    svr = cross_val_predict(alg3, X, y=Y, cv=CV, n_jobs=-1)
    X = pd.DataFrame(X, index=Y.index, columns=X_unscaled.columns.values.tolist())
    preds_by_alg = X.assign(TrueY = Y, kNN = knn, DTree = dtr, SVR = svr)
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

def ext_test_compare(X, Y, alg1, alg2, alg3, csv_name):
    """
    X : dataframe that includes all training data
    Y : series with labels for training data
    alg1 : optimized learner 1
    alg2 : optimized learner 2
    alg3 : optimized learner 3
    csv_name : string containing the train set, nuc subset, and parameter being 
               predicted for naming purposes

    Returns
    -------
    *.csv : csv file with each alg's predictions compared to ground truth
    
    """
    testpath = 'learn/pkl_trainsets/2jul2018/2jul2018_testset1_'
    test_pkl = testpath + 'nucs_fissact_not-scaled.pkl'
    testXY = pd.read_pickle(test_pkl)
    testXY.reset_index(inplace=True, drop=True) 
    testX, rY, cY, eY, bY = splitXY(testXY)
    if 'reactor' in str(csv_name):
        testY = rY
    elif 'cooling' in str(csv_name):
        testY = cY
    elif 'enrichment' in str(csv_name):
        testY = eY
    else:
        testY = bY
    # fit w data
    alg1.fit(X, Y)
    alg2.fit(X, Y)
    alg3.fit(X, Y)
    # make predictions
    knn = alg1.predict(testX)
    dtr = alg2.predict(testX)
    svr = alg3.predict(testX)
    alg_preds = pd.DataFrame({'TrueY': testY, 'kNN': knn, 
                              'DTree': dtr, 'SVR': svr}, 
                              index=testY.index)
    alg_preds.to_csv(csv_name + '_ext_test_compare.csv')
    return
