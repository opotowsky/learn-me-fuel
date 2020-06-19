#! /usr/bin/env python3

import glob
import numpy as np
import pandas as pd

def calc_errors(pred_df, true_lbls):
    """
    Given a dataframe containing predictions and log-likelihood value,
    calculates absolute error between predictions and ground truth (or boolean
    where applicable)

    Parameters
    ----------
    pred_df : dataframe with ground truth and predicted labels
    true_lbls : list of ground truth column labels 
    
    Returns
    -------
    pred_df : dataframe with ground truth, predictions, and errors between the 
              two
    
    """
    pred_lbls = ["pred_" + s for s in true_lbls] 
    for true, pred in zip(true_lbls, pred_lbls):
        if 'Reactor' in true:
            col_name = true + '_Score'
            pred_df[col_name] = np.where(pred_df.loc[:, true] == pred_df.loc[:, pred], True, False)
        else: 
            col_name = true + '_Error'
            pred_df[col_name] = np.abs(pred_df.loc[:, true]  - pred_df.loc[:, pred])
    return pred_df

def main():
    """
    
    
    """

    out_dir = './results/'
    df_csvs = glob.glob(out_dir + '*.csv')
    df_csvs.sort()
    pred_df = pd.DataFrame()
    for csv in df_csvs:
        chunk = pd.read_csv(csv)
        pred_df = pred_df.append(chunk)
    
    # copied from mll_calc.py for now
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    pred_df = calc_errors(pred_df, lbls)
    
    fname = args.outfile + '.csv'
    pred_df.to_csv(fname)

    return

if __name__ == "__main__":
    main()
