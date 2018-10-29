#! /usr/bin/env python3

import pandas as pd

def get_algparams(prediction_name):
    """
    Gives the optimized algorithm hyperparamters with respect to reactor
    prediction label 

    Parameters
    ----------
    prediction_name : string of the reactor parameter to be predicted 

    Returns
    -------
    k : integer, number of nearest neighbors in kNN
    depth : integer, maximum depth of decision trees
    feats : integer, maximum number of features in decision trees
    g : float, gamma parameter in support vectors
    c : float, C parameter in support vectors
    """
    name = prediction_name
    if name == 'cooling':
        k = 3 #7
        depth = 50 #50, 12
        feats = 25 #36, 47 
        g = 0.06 #0.2
        c = 50000 #200, 75000
    elif name == 'enrichment': 
        k = 7 #8
        depth = 50 #53, 38
        feats = 25 #33, 16 
        g = 0.8 #0.2
        c = 25000 #420
    elif name == 'burnup':
        k = 7 #4
        depth = 50 #50, 78
        feats = 25 #23, 42 
        g = 0.25 #0.025
        c = 42000 #105
    else: # name == 'reactor'
        k = 3 #1, 2, or 12
        depth = 50 #50, 97
        feats = 25 # 37, 37 
        g = 0.07 #0.2
        c = 1000 #220
    return k, depth, feats, g, c

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

