#! /usr/bin/env python3

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

