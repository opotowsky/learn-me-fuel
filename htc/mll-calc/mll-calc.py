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
    pred_lbls.append(ll_name)
    pred_ll = pred_ll.loc[:, pred_lbls]
    
    return pred_ll

def main():
    """
    """
    
    parser = argparse.ArgumentParser(description='Performs maximum likelihood calculations for reactor parameter prediction.')
    parser.add_argument('unc', metavar='simulation-uncertainty', 
                        help='value of simulation uncertainty (in fraction) to apply to likelihood calculations')
    parser.add_argument('-r', '--ratios', action='store_true', default=False, 
                        help='compute isotopic ratios instead of using concentrations (default)')
    args = parser.parse_args()
    
    pklfile = '~/prep-pkls/nucmoles_opusupdate_aug2019/not-scaled_15nuc.pkl'
    XY = pd.read_pickle(pklfile)
    XY.reset_index(inplace=True, drop=True)
    if 'total' in XY.columns:
        XY.drop('total', axis=1, inplace=True)
    XY = XY.loc[XY['Burnup'] > 0]
    # small db for testing code
    XY = XY.sample(50)
    
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    if args.ratios == True:
        XY = ratios(XY, lbls)

    unc = float(args.unc)
    pred_df = pd.DataFrame()
    for sim_idx, row in XY.iterrows():
        test_sample = XY.loc[XY.index == sim_idx].drop(lbls, axis=1)
        test_answer = XY.loc[XY.index == sim_idx, lbls]
        trainXY = XY.drop(sim_idx)#copy()
        pred_ll = get_pred(trainXY, test_sample, unc, lbls)
        if pred_df.empty:
            pred_df = pd.DataFrame(columns = pred_ll.columns.to_list())
        pred_df = pred_df.append(pred_ll)
    
    fname = 'test_mll'
    pred_pkl = fname + '.pkl'
    pickle.dump(pred_df, open(pred_pkl, 'wb'))
    pred_df.to_csv(fname + '.csv')
    compression_opts = dict(method='zip', archive_name='fname' + '_comp.csv')
    pred_df.to_csv(fname + '.zip', compression=compression_opts)

    return

if __name__ == "__main__":
    main()
