#! /usr/bin/env python3

import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, explained_variance_score, mean_absolute_error

def like_calc(y_sim, y_mes, std):
    """
    ddddd

    Parameters
    ----------
    y_sim : 
    y_mes : 
    std : 

    Returns
    -------
    like: 

    """
    like = np.prod(stats.norm.pdf(y_sim, loc=y_mes, scale=std))
    return like

def ll_calc(y_sim, y_mes, std):
    """
    ddddd

    Parameters
    ----------
    y_sim : 
    y_mes : 
    std : 

    Returns
    -------
    like: 

    """
    ll = np.sum(stats.norm.logpdf(y_sim, loc=y_mes, scale=std))
    return ll

def unc_calc(y_sim, y_mes, sim_unc_sq, mes_unc_sq):
    """
    ddddd

    Parameters
    ----------
    y_sim : 
    y_mes : 
    sim_unc_sq : 
    mes_unc_sq : 

    Returns
    -------
    ll_unc : 

    """
    unc = ((y_sim - y_mes) / sim_unc_sq)**2 * (sim_unc_sq + mes_unc_sq)
    unc.replace([np.inf, -np.inf], 0, inplace=True)
    unc.fillna(0, inplace=True)
    ll_unc = np.sqrt(unc.sum(axis=1))
    return ll_unc

def ratios(XY, labels):
    """
    dddd

    Parameters
    ----------
    XY : 
    labels : 

    Returns
    -------
    XY_ratios : 

    """

    XY_ratios = XY.loc[:, labels].copy()
    
    #cs137/cs133
    XY_ratios['cs137/cs133'] = XY['cs137'] / XY['cs133']
    #cs134/cs137
    XY_ratios['cs134/cs137'] = XY['cs134'] / XY['cs137']
    #cs135/cs137
    XY_ratios['cs135/cs137'] = XY['cs135'] / XY['cs137']
    #ba136/ba138
    XY_ratios['ba136/ba138'] = XY['ba136'] / XY['ba138']
    #sm150/sm149
    XY_ratios['sm150/sm149'] = XY['sm150'] / XY['sm149']
    #sm152/sm149
    XY_ratios['sm152/sm149'] = XY['sm152'] / XY['sm149']
    #eu154/eu153
    XY_ratios['eu154/eu153'] = XY['eu154'] / XY['eu153']
    #pu240/pu239
    XY_ratios['pu240/pu239'] = XY['pu240'] / XY['pu239']
    #pu241/pu239
    XY_ratios['pu241/pu239'] = XY['pu241'] / XY['pu239']
    #pu242/pu239
    XY_ratios['pu242/pu239'] = XY['pu242'] / XY['pu239']
    
    XY_ratios.replace([np.inf, -np.inf], 0, inplace=True)
    XY_ratios.fillna(0, inplace = True)
    return XY_ratios

def get_pred(XY, test_sample, unc, lbls):
    """
    dddd

    Parameters
    ----------
    XY : 
    test_sample :
    unc : 
    lbls : 

    Returns
    -------
    XY : 
    
    """
    ll_name = 'LogLikelihood_' + str(unc)
    unc_name = 'LLUncertainty_' + str(unc)
    X = XY.drop(lbls, axis=1).copy()
    
    XY[ll_name] = X.apply(lambda row: ll_calc(row, test_sample, unc*row), axis=1)
    #XY[unc_name] = X.apply(lambda row: unc_calc(row, test_sample, (unc*row)**2, (unc*test_sample)**2), axis=1)
    
    #max_ll = XY[ll_name].max()
    max_idx = XY[ll_name].idxmax()
    pred_ll = XY.loc[XY.index == max_idx]#.drop(ll_name, axis=1)
    
    pred_lbls = ["Pred_" + s for s in lbls] 
    pred_ll.rename(columns=dict(zip(lbls, pred_lbls)), inplace=True)
    pred_lbls.append(ll_name)#.append('Pred_idx')
    pred_ll = pred_ll.loc[:, pred_lbls]
    
    return pred_ll, pred_lbls

def calc_errors(pred_df, true_lbls, pred_lbls):
    """
    dddd

    Parameters
    ----------
    pred_df : 
    true_lbls : 
    pred_lbls : 
    
    Returns
    -------
    pred_df : 
    
    """
    for true, pred in zip(true_lbls, pred_lbls):
        if 'Reactor' in true:
            col_name = true + '_Score'
            pred_df[col_name] = np.where(pred_df.loc[:, true] == pred_df.loc[:, pred], True, False)
        else: 
            col_name = true + '_Error'
            pred_df[col_name] = np.abs(pred_df.loc[:, true]  - pred_df.loc[:, pred])

    return pred_df

def loop_db(train, test, unc, lbls):
    """
    dddd

    Parameters
    ----------
    train : 
    test : 
    unc : 
    lbls : 

    """
    pred_df = pd.DataFrame()
    for sim_idx, row in test.iterrows():
        test_sample = test.loc[test.index == sim_idx].drop(lbls, axis=1)
        test_answer = test.loc[test.index == sim_idx, lbls]
        pred_ll, pred_lbls = get_pred(train, test_sample, unc, lbls)
        if pred_df.empty:
            pred_df = pd.DataFrame(columns = pred_ll.columns.to_list())
        pred_df = pred_df.append(pred_ll)
    pred_df = pd.concat([test.loc[:, lbls].rename_axis('sim_idx').reset_index(), 
                         pred_df.rename_axis('pred_idx').reset_index()
                         ], axis=1)
    
    pred_df = calc_errors(pred_df, lbls, pred_lbls)

    fname = 'test_mll'
    pred_pkl = fname + '.pkl'
    pickle.dump(pred_df, open(pred_pkl, 'wb'))
    pred_df.to_csv(fname + '.csv')
    compression_opts = dict(method='zip', archive_name='fname' + '_comp.csv')
    pred_df.to_csv(fname + '.zip', compression=compression_opts)
    
    return

def main():
    """
    """
    
    parser = argparse.ArgumentParser(description='Performs maximum likelihood calculations for reactor parameter prediction.')
    parser.add_argument('unc', metavar='simulation-uncertainty', 
                        help='value of simulation uncertainty (in fraction) to apply to likelihood calculations')
    parser.add_argument('-e', '--ext-test', action='store_true', default=False, 
                        help='execute script with external testing set instead of training set evaluation (default)')
    parser.add_argument('-r', '--ratios', action='store_true', default=False, 
                        help='compute isotopic ratios instead of using concentrations (default)')
    args = parser.parse_args()
    
    # hard-coded filepaths
    trainfile = '~/prep-pkls/nucmoles_opusupdate_aug2019/not-scaled_15nuc.pkl'
    sfcompofile = '~/sfcompo/format_clean/sfcompo_format.pkl'

    # training set
    train = pd.read_pickle(trainfile)
    train.reset_index(inplace=True, drop=True)
    if 'total' in train.columns:
        train.drop('total', axis=1, inplace=True)
    train = train.loc[XY['Burnup'] > 0]

    # small db for testing code
    train = train.sample(50)

    # testing set
    if args.ext-test == True:
        test = pd.read_pickle(sfcompofile)
        #test.reset_index(inplace=True, drop=True)
    else: 
        test = train
        

    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    if args.ratios == True:
        train = ratios(train, lbls)
    unc = float(args.unc)

    loop_db(train, test, unc, lbls)

    return

if __name__ == "__main__":
    main()
