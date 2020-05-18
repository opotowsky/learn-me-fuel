#! /usr/bin/env python3

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, explained_variance_score, mean_absolute_error

def like_calc(y_sim, y_mes, std):
    """
    Given a simulated entry with uncertainty and a test entry, calculates the
    likelihood that they are the same. 
    
    Parameters
    ----------
    y_sim : series of nuclide measurements for simulated entry
    y_mes : series of nuclide measurements for test ("measured") entry
    std : standard deviation for each entry in y_sim

    Returns
    -------
    like: likelihood that the test entry is the simulated entry

    """
    like = np.prod(stats.norm.pdf(y_sim, loc=y_mes, scale=std))
    return like

def ll_calc(y_sim, y_mes, std):
    """
    Given a simulated entry with uncertainty and a test entry, calculates the
    log-likelihood that they are the same. 

    Parameters
    ----------
    y_sim : series of nuclide measurements for simulated entry
    y_mes : series of nuclide measurements for test ("measured") entry
    std : standard deviation for each entry in y_sim

    Returns
    -------
    ll: log-likelihood that the test entry is the simulated entry

    """
    ll = np.sum(stats.norm.logpdf(y_sim, loc=y_mes, scale=std))
    return ll

def unc_calc(y_sim, y_mes, sim_unc_sq, mes_unc_sq):
    """
    Given a simulated entry and a test entry with uniform uncertainty,
    calculates the uncertainty in the log-likelihood calculation. 

    Parameters
    ----------
    y_sim : series of nuclide measurements for simulated entry
    y_mes : series of nuclide measurements for test ("measured") entry
    sim_unc_sq : float of squared uniform uncertainty for each entry in y_sim
    mes_unc_sq : float of squared uniform uncertainty for each entry in y_mes

    Returns
    -------
    ll_unc: uncertainty of the log-likelihood calculation

    """
    unc = ((y_sim - y_mes) / sim_unc_sq)**2 * (sim_unc_sq + mes_unc_sq)
    unc.replace([np.inf, -np.inf], 0, inplace=True)
    unc.fillna(0, inplace=True)
    ll_unc = np.sqrt(unc.sum(axis=1))
    return ll_unc

def ratios(XY, ratio_list, labels):
    """
    Given a dataframe with entries (rows) that contain nuclide measurements and
    some labels, calculate the predetermined ratios of the measurements.

    Parameters
    ----------
    XY : dataframe of spent fuel entries containing nuclide measurements and 
         their labels
    ratio_list : list of ratios desired
    labels : list of label titles in the dataframe

    Returns
    -------
    XY_ratios : dataframe of spent fuel entries containing nuclide measurement 
                ratios and their labels

    """

    XY_ratios = XY.loc[:, labels].copy()
    for ratio in ratio_list: 
        nucs = ratio.split('/')
        XY_ratios[ratio] = XY[nucs[0]] / XY[nucs[1]]
    XY_ratios.replace([np.inf, -np.inf], 0, inplace=True)
    XY_ratios.fillna(0, inplace = True)
    # reorganize columns
    cols = ratio_list + labels
    XY_ratios = XY_ratios[cols]
    return XY_ratios

def get_pred(XY, test_sample, unc, lbls):
    """
    Given a database of spent fuel entries and a test sample (nuclide
    measurements only), calculates the log-likelihood (and LL-uncertainty) of
    that sample against every database entry.  Determines the max LL, and
    therefore the corresponding prediction in the database.  Returns that
    prediction as a single row dataframe.

    Parameters
    ----------
    XY : dataframe with nuclide measurements and reactor parameters
    test_sample : series of a sample to be predicted (nuclide measurements only)
    unc : float that represents the simulation uncertainty in nuclide measurements
    lbls : list of reactor parameters to be predicted

    Returns
    -------
    pred_ll : dataframe with single row of prediction (predicted labels only) 
              and its log-likelihood
    pred_lbls : list of predicted label titles

    """
    ll_name = 'LogLikelihood_' + str(unc)
    unc_name = 'LLUncertainty_' + str(unc)
    X = XY.drop(lbls, axis=1).copy()
    
    ##### need to test that the rows are matching isos properly, have test columns set sorted for now ####
    #### might need to do in in the function that calls this one

    XY[ll_name] = X.apply(lambda row: ll_calc(row, test_sample, unc*row), axis=1)
    #XY[unc_name] = X.apply(lambda row: unc_calc(row, test_sample, (unc*row)**2, (unc*test_sample)**2), axis=1)
    
    #max_ll = XY[ll_name].max()
    #### TO DO ####
    # will there ever be more than one max? unlikely but maybe should see about 
    # handling this just in case there are somehow duplicate entries. script 
    # will likely fail
    max_idx = XY[ll_name].idxmax()
    #### END TO DO ####
    pred_ll = XY.loc[XY.index == max_idx].copy()
    # need to delete likelihood column so next test sample can be calculated
    XY.drop(ll_name, axis=1, inplace=True)
    
    pred_lbls = ["pred_" + s for s in lbls] 
    pred_ll.rename(columns=dict(zip(lbls, pred_lbls)), inplace=True)
    pred_lbls.append(ll_name)
    pred_ll = pred_ll.loc[:, pred_lbls]

    return pred_ll, pred_lbls

def calc_errors(pred_df, true_lbls, pred_lbls):
    """
    Given a dataframe containing predictions and log-likelihood value,
    calculates absolute error between predictions and ground truth (or boolean
    where applicable)

    Parameters
    ----------
    pred_df : dataframe with ground truth and predicted labels
    true_lbls : list of ground truth column labels 
    pred_lbls : list of prediction column labels
    
    Returns
    -------
    pred_df : dataframe with ground truth, predictions, and errors between the 
              two
    
    """
    #### TO DO ####
    # can I separate true from pred here so pred_lbls doesn't need to be 
    # passed around a bunch?
    # This also assumes two lists match feature ordering, should check for 
    # that somewhere in-script
    #### END TO DO ####
    for true, pred in zip(true_lbls, pred_lbls):
        if 'Reactor' in true:
            col_name = true + '_Score'
            pred_df[col_name] = np.where(pred_df.loc[:, true] == pred_df.loc[:, pred], True, False)
        else: 
            col_name = true + '_Error'
            pred_df[col_name] = np.abs(pred_df.loc[:, true]  - pred_df.loc[:, pred])

    return pred_df

def mll_testset(XY, test, unc, lbls):
    """
    Given a database of spent fuel entries containing a nuclide vector and the
    reactor operation parameters, and an equally formatted database of test
    cases to predict, this function loops through the test database to perform
    a series of predictions.  It first formats the test sample for prediction,
    then gathers all the predictions from the test database entries

    Parameters
    ----------
    XY : dataframe with nuclide measurements and reactor parameters
    test : dataframe with test cases to predict in same format as train
    unc : float that represents the simulation uncertainty in nuclide measurements
    lbls : list of reactor parameters to be predicted

    Returns
    -------
    pred_df : dataframe with ground truth and predictions
    pred_lbls : list of predicted label titles
    
    """
    pred_df = pd.DataFrame()
    loov = XY.equals(test)
    for sim_idx, row in test.iterrows():
        test_sample = row.drop(lbls)
        test_answer = row[lbls]
        if loov:
            pred_ll, pred_lbls = get_pred(XY.drop(sim_idx), test_sample, unc, lbls)
            #XY.loc[sim_idx] = row # don't need since drop is now not saving df inplace
        else:
            pred_ll, pred_lbls = get_pred(XY, test_sample, unc, lbls)
        if pred_df.empty:
            #### TO DO ####
            # If pred_lbls stays, can make this columns = line shorter
            pred_df = pd.DataFrame(columns = pred_ll.columns.to_list())
            #### END TO DO ####
        pred_df = pred_df.append(pred_ll)
    pred_df = pd.concat([test.loc[:, lbls].rename_axis('sim_idx').reset_index(), 
                         pred_df.rename_axis('pred_idx').reset_index()
                         ], axis=1)
    
    return pred_df, pred_lbls

def parse_args(args):
    parser = argparse.ArgumentParser(description='Performs maximum likelihood calculations for reactor parameter prediction.')
    
    # hard-coded filepaths
    dbfile = '~/prep-pkls/nucmoles_opusupdate_aug2019/not-scaled_15nuc.pkl'
    sfcompofile = '~/sfcompo/format_clean/sfcompo_formatted.pkl'
    
    parser.add_argument('train', metavar='training-set', 
                        nargs='?', default=dbfile, 
                        help='specify file path to a training set to override default path in script')
    parser.add_argument('test', metavar='testing-set', 
                        nargs='?', default=sfcompofile,
                        help='specify file path to a testing set to override default path in script')
    parser.add_argument('unc', metavar='simulation-uncertainty', 
                        nargs='?', default=0.05, type=float,
                        help='value of simulation uncertainty (in fraction) to apply to likelihood calculations')
    parser.add_argument('-e', '--ext_test', action='store_true', default=False, 
                        help='execute script with external testing set instead of training set evaluation (default)')
    parser.add_argument('-r', '--ratios', action='store_true', default=False, 
                        help='compute isotopic ratios instead of using concentrations (default)')
    
    return parser.parse_args(args)

def main():
    """
    Given a database of spent fuel entries (containing nuclide measurements and
    labels of reactor operation parameters of interest for prediction) and a
    testing database containing spent fuel entries formatted in the same way,
    this script calculates the maximum log-likelihood of each test sample
    against the database for a prediction. The errors of those predictions are
    then calculated and saved as a CSV file.
    
    """
    
    args = parse_args(sys.argv[1:])

    # training set
    XY = pd.read_pickle(args.train)
    XY.reset_index(inplace=True, drop=True)
    if 'total' in XY.columns:
        XY.drop('total', axis=1, inplace=True)
    XY = XY.loc[XY['Burnup'] > 0]

    #### TO REMOVE ####
    # small db for testing code
    #XY = XY.sample(50)
    XY = XY.head(25)
    #### END REMOVE ####

    # testing set
    if args.ext_test == True:
        test = pd.read_pickle(args.test)
        # order of columns must match
        if XY.columns.tolist() != test.columns.tolist():
            if sorted(XY.columns.tolist()) == sorted(test.columns.tolist()):
                test = test[XY.columns]
            else:
                sys.exit("Feature sets are different")
        #### TO REMOVE ####
        # small db for testing code
        test = test.sample(1)
        #### END REMOVE ####
    else: 
        test = XY.copy()
        
    tamu_list = ['cs137/cs133', 'cs134/cs137', 'cs135/cs137', 'ba136/ba138', 
                 'sm150/sm149', 'sm152/sm149', 'eu154/eu153', 'pu240/pu239', 
                 'pu241/pu239', 'pu242/pu239'
                 ]
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    if args.ratios == True:
        #### TO DO ####
        # This list (in future, lists) will correspond with the dbs that are 
        # imported, currently hard-coded filepaths above. This will need to be
        # moved outside the script and handled differently via some kind of 
        # config file
        ratio_list = tamu_list
        #### END TO DO ####
        XY = ratios(XY, ratio_list, lbls)
        test = ratios(test, ratio_list, lbls)
    
    unc = float(args.unc)
    pred_df, pred_lbls = mll_testset(XY, test, unc, lbls)
    pred_df = calc_errors(pred_df, lbls, pred_lbls)

    # testing multiple formats in case the DBs get big enough for this to matter
    fname = 'test_mll'
    #pred_pkl = fname + '.pkl'
    #pickle.dump(pred_df, open(pred_pkl, 'wb'))
    pred_df.to_csv(fname + '.csv')
    #compression_opts = dict(method='zip', archive_name='fname' + '_comp.csv')
    #pred_df.to_csv(fname + '.zip', compression=compression_opts)

    return

if __name__ == "__main__":
    main()
