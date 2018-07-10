#! /usr/bin/env python3

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, StratifiedKFold
import pandas as pd
import numpy as np

def errors_and_scores(trainX, Y, knn_init, rr_init, svr_init, rxtr_pred, scores, CV, subset, trainset):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    """
    # init/empty the lists
    knn_scores = []
    rr_scores = []
    svr_scores = []
    for alg in ('knn', 'rr', 'svr'):
        # get init'd learner
        if alg == 'knn':
            alg_pred = knn_init
        elif alg == 'rr':
            alg_pred = rr_init
        else:
            alg_pred = svr_init
        # cross valiation to obtain scores
        cv_info = cross_validate(alg_pred, trainX, Y, scoring=scores, cv=CV, return_train_score=False)
        df = pd.DataFrame(cv_info)
        if rxtr_pred is not 'reactor':
            # to get MSE -> RMSE
            test_mse = df['test_neg_mean_squared_error']
            rmse = lambda x : -1 * np.sqrt(-1*x)
            df['test_neg_rmse'] = rmse(test_mse)
        # for concatting since it's finnicky
        if alg == 'knn':
            df['algorithm'] = 'knn'
            knn_scores = df
        elif alg == 'rr':
            df['algorithm'] = 'rr'
            rr_scores = df
        else:
            df['algorithm'] = 'svr'
            svr_scores = df
    cv_results = [knn_scores, rr_scores, svr_scores]
    df = pd.concat(cv_results)
    df.to_csv('trainset_' + trainset + '_' + subset + '_' + rxtr_pred + '_scores.csv')
    return

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

    """

    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    dfX = dfXY.drop(lbls, axis=1)
    if 'total' in dfX.columns:
        data.drop('total', axis=1, inplace=True)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY

def main():
    """
    Given training data, this script trains and tracks each prediction for
    several algorithms and saves the predictions and ground truth to a CSV file
    """
    # Parameters for the training and predictions
    CV = 10
    # The hand-picked numbers are based on the dayman test set validation curves
    k = 13
    a = 100
    g = 0.001
    c = 10000

    pkl_base = './pkl_trainsets/2jul2018/2jul2018_trainset'
    for trainset in ('1', '2'):
        for subset in ('fiss', 'act', 'fissact', 'all'):
            pkl = pkl_base + trainset + '_nucs_' + subset + '_not-scaled.pkl'
            trainXY = pd.read_pickle(pkl)
            trainX, rY, cY, eY, bY = splitXY(trainXY)
            if subset == 'all':
                top_n = 100
                nuc_set = top_nucs(trainX, top_n)
                trainX = filter_nucs(trainX, nuc_set, top_n)
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
                
                # initialize a learner
                if Y is not 'r':
                    scores = ['r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error']
                    kfold = KFold(n_splits=CV, shuffle=True)
                    knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    rr_init = Ridge(alpha=a)
                    svr_init = SVR(gamma=g, C=c)
                else:
                    scores = ['accuracy', ]
                    kfold = StratifiedKFold(n_splits=CV, shuffle=True)
                    knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
                    rr_init = RidgeClassifier(alpha=a, class_weight='balanced')
                    svr_init = SVC(gamma=g, C=c, class_weight='balanced')
                
                # make predictions
                knn = cross_val_predict(knn_init, trainX, y=trainY, cv=kfold)
                rr = cross_val_predict(rr_init, trainX, y=trainY, cv=kfold)
                svr = cross_val_predict(svr_init, trainX, y=trainY, cv=kfold)
                preds_by_alg = pd.DataFrame({'TrueY': trainY, 'kNN': knn, 
                                             'Ridge': rr, 'SVR': svr}, 
                                             index=trainY.index)
                preds_by_alg.to_csv('trainset_' + trainset + '_' + subset + '_' + parameter + '_predictions.csv')
                
                # calculate errors and scores
                errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, parameter, scores, kfold, subset, trainset)

                # compare fit to Dayman test set
                testpkl_base = './pkl_trainsets/2jul2018/2jul2018_testset1'
                for subset in ('fiss', 'act', 'fissact', 'all'):
                    pkl = testpkl_base + '_nucs_' + subset + '_not-scaled.pkl'
            trainXY = pd.read_pickle(pkl)
            trainX, rY, cY, eY, bY = splitXY(trainXY)
        # fit w data
        knn_init.fit(trainX, trainY)
        rr_init.fit(trainX, trainY)
        svr_init.fit(trainX, trainY)
        # make predictions
        knn = knn_init.predict(testX)
        rr = rr_init.predict(testX)
        svr = svr_init.predict(testX)
        preds_by_alg = pd.DataFrame({'TrueY': testY, 'kNN': knn, 
                                     'Ridge': rr, 'SVR': svr}, 
                                    index=testY.index)
        preds_by_alg.to_csv('lowburn_' + parameter + '_predictions.csv')
        # calculate errors and scores
        errors_and_scores(testY, knn, rr, svr, parameter)
    
    ### Errors and scores code from lowburn.py
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    """
    cols = ['r2 Score', 'Explained Variance', 'Negative MAE', 'Negative RMSE']
    idx = ['kNN', 'Ridge', 'SVR']
    # init empty lists
    knn_scores = []
    rr_scores = []
    svr_scores = []
    for alg in ('knn', 'rr', 'svr'):
        # get pred list
        if alg == 'knn':
            alg_pred = knn
        elif alg == 'rr':
            alg_pred = rr
        else:
            alg_pred = svr
        
        # 4 calculations of various 'scores':
        r2 = r2_score(testY, alg_pred)
        exp_var = explained_variance_score(testY, alg_pred)
        mae = -1 * mean_absolute_error(testY, alg_pred)
        rmse =-1 * np.sqrt(mean_squared_error(testY, alg_pred))
        
        scores = [r2, exp_var, mae, rmse]
        if alg == 'knn':
            knn_scores = scores
        elif alg == 'rr':
            rr_scores = scores
        else:
            svr_scores = scores
    df = pd.DataFrame([knn_scores, rr_scores, svr_scores], index=idx, columns=cols)
    df.to_csv('lowburn_' + rxtr_pred + '_scores.csv')
    return

    return

if __name__ == "__main__":
    main()
