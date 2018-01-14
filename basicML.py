#! /usr/bin/env python

from __future__ import print_function
from __future__ import division
from manual_preds import manual_train_and_predict
from info_reduc import random_error
from auto_preds import auto_train_and_predict
from sklearn.preprocessing import scale
import pickle
import math
import numpy as np
import pandas as pd
import glob
import os

class LearnSet(object):
    """
    A set of parameters (i.e., features and labels) for a machine learning 
    algorithm, each in the format of a pandas dataframe
    """

    def __init__(self, nuc_concs, burnup):#reactor, enrichment, burnup):
        self.nuc_concs = nuc_concs
        #self.reactor = reactor
        #self.enrichment = enrichment
        self.burnup = burnup


###################################################
# TODO: Leaving the following global for now; fix!#
###################################################

# Info for labeling the simulation values in the training set
pwrburn = (600, 1550, 2500, 3450, 4400, 5350, 6300, 7250, 8200, 9150, 10100, 
           11050, 12000, 12950, 13900, 14850, 15800, 16750, 17700
           )
bwrburn = (600, 1290, 1980, 2670, 3360, 4050, 4740, 5430, 6120, 6810, 7500, 
           8190, 8880, 9570, 10260, 10950, 11640, 12330
           )
phwrburn = (600, 1290, 1980, 2670, 3360, 4050, 4740, 5430, 6120, 6810, 7500, 
            8190, 8880, 9570, 10260, 10950, 11640, 12330
            )
o_rxtrs = ('ce14x14', 'ce16x16', 'w14x14', 'w15x15', 'w17x17', 's14x14', 
           'vver440', 'vver440_3.82', 'vver440_4.25', 'vver440_4.38', 
           'vver1000', 'ge7x7-0', 'ge8x8-1', 'ge9x9-2', 'ge10x10-8', 
           'abb8x8-1', 'atrium9x9-9', 'svea64-1', 'svea100-0', 'candu28', 
           'candu37'
           )
enrich =  (2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 3.6, 3.82, 4.25, 4.38, 2.8, 2.9, 
           2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 0.711, 0.711
           )
train_label = {'ReactorType': ['pwr']*11 + ['bwr']*8 + ['phwr']*2,
               'OrigenReactor': o_rxtrs,
               'Enrichment': enrich,
               'Burnup': [pwrburn]*11 + [bwrburn]*8 + [phwrburn]*2,
               'CoolingInts': [(0.000694, 7, 30, 365.25)]*21
               }

# Info for labeling the simulated/expected values in the testing set
t_burns = ((1400, 5000, 11000), (5000, 6120), (1700, 8700, 17000),
           (8700, 9150), (8700, 9150), (2000, 7200, 10800),
           (7200, 8800), (7200, 8800)
           )
cool1 = (0.000694, 7, 30, 365.25) #1 min, 1 week, 1 month, 1 year in days
cool2 = (0.002082, 9, 730.5) #3 min, 9 days, 2 years in days
cool3 = (7, 9) #7 and 9 days
t_o_rxtrs = ('candu28_0', 'candu28_1', 'ce16x16_2', 'ce16x16_3', 'ce16x16_4', 
             'ge7x7-0_5','ge7x7-0_6', 'ge7x7-0_7'
             )
t_enrich =  (0.711, 0.711, 2.8, 2.8, 3.1, 2.9, 2.9, 3.2)
test_label = {'ReactorType': ['phwr']*2 + ['pwr']*3 + ['bwr']*3,
              'OrigenReactor': t_o_rxtrs,
              'Enrichment': t_enrich,
              'Burnup': t_burns, 
              'CoolingInts': [cool1, cool2, cool1, cool2, cool3, cool1, cool2, cool3]
              }

def format_df(filename):
    """
    This takes a csv file and reads the data in as a dataframe.

    Parameters
    ----------
    filename : str of simulation output in a csv file

    Returns
    -------
    data : pandas dataframe containing csv entries

    """
    
    data = pd.read_csv(filename).T
    data.columns = data.iloc[0]
    data.drop(data.index[0], inplace=True)
    return data

def get_labels(filename, rxtrs):
    """
    This takes a filename and a dict with all simulation parameters, and 
    searches for the entries relevant to the given simulation (file).

    Parameters
    ----------
    filename : str of simulation output in a csv file
    rxtrs : dict of a data set detailing simulation parameters in ORIGEN
    
    Returns
    -------
    rxtr_info : dict of all the labels for a given simulation data set

    """
    
    tail, _ = os.path.splitext(os.path.basename(filename))
    i = rxtrs['OrigenReactor'].index(tail)
    rxtr_info = {'ReactorType': rxtrs['ReactorType'][i], 
                 'Enrichment': rxtrs['Enrichment'][i], 
                 'Burnup': rxtrs['Burnup'][i], 
                 'CoolingInts': rxtrs['CoolingInts'][i]
                 }
    return rxtr_info

def label_data(label, data):
    """
    Takes the labels for and a dataframe of the simulation results; 
    adds these labels as additional columns to the dataframe.

    Parameters
    ----------
    label : dict representing the labels for a simulation
    data : dataframe of simulation results

    Returns
    -------
    data : dataframe of simulation results + label entries in columns

    """
    
    col = len(data.columns)
    data.insert(loc = col, column = 'ReactorType', value = label['ReactorType'])
    data.insert(loc = col+1, column = 'Enrichment', value = label['Enrichment'])
    burnup = burnup_label(label['Burnup'], label['CoolingInts'])
    data.insert(loc = col+2, column = 'Burnup', value = burnup)
    return data

def burnup_label(burn_steps, cooling_ints):
    """
    Takes the burnup steps and cooling intervals for each case within the 
    simulation and creates a list of the burnup of the irradiated and cooled/ 
    decayed fuels; returns a list to be added as the burnup label to the main 
    dataframe.

    Parameters
    ----------
    burn_steps : list of the steps of burnup from the simulation parameters
    cooling_ints : list of the cooling intervals from the simulation parameters

    Returns
    -------
    burnup_list : list of burnups to be applied as a label for a given simulation

    """
    
    num_cases = len(burn_steps)
    steps_per_case = len(cooling_ints) + 2
    burnup_list = [0, ]
    for case in range(0, num_cases):
        for step in range(0, steps_per_case):
            if (case == 0 and step == 0):
                continue
            elif (case > 0 and step == 0):
                burn_step = burn_steps[case-1]
                burnup_list.append(burn_step)
            else:
                burn_step = burn_steps[case]
                burnup_list.append(burn_step)
    return burnup_list

def dataframeXY(all_files, rxtr_label):
    """"
    Takes the glob of files in a directory as well as the dict of labels and 
    produces a dataframe that has both the data features (X) and labeled data (Y).

    Parameters
    ----------
    all_files : list of str holding all simulation file names in a directory
    rxtr_label : dict holding all parameters for all simulations in a directory

    Returns
    -------
    dfXY : dataframe that has all features and labels for all simulations in a 
           directory

    """

    all_data = []
    for f in all_files:
        data = format_df(f)
        labels = get_labels(f, rxtr_label)
        labeled = label_data(labels, data)
        all_data.append(labeled)
    dfXY = pd.concat(all_data)
    ##FILTERING STUFFS##
    # Delete sim columns
    # Need better way to know when the nuclide columns start (6 for now)
    # Prob will just search for column idx that starts with str(1)?
    cols = len(dfXY.columns)
    dfXY = dfXY.iloc[:, 6:cols]
    # Filter out 0 burnups so MAPE can be calc'd
    dfXY = dfXY.loc[dfXY.Burnup > 0, :]
    return dfXY

def top_nucs(dfXY, top_n):
    """
    loops through the rows of a dataframe and keeps the top_n nuclides 
    (by concentration) from each row
    
    Parameters
    ----------
    dfXY : dataframe of nuclide concentrations + labels
    top_n : number of nuclides to sort and filter by

    Returns
    -------
    nuc_set : set of the top_n nucs as determined 

    """
    
    x = len(dfXY.columns)-3
    dfX = dfXY.iloc[:, 0:x]
    # Get a set of top n nucs from each row (instance)
    nuc_set = set()
    for case, conc in dfX.iterrows():
        top_n_series = conc.sort_values(ascending=False)[:top_n]
        nuc_list = list(top_n_series.index.values)
        nuc_set.update(nuc_list)
    return nuc_set

def filter_nucs(df, nuc_set, top_n):
    """
    for each instance (row), keep only top 200 values, replace rest with 0
    
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
    produces four different dataframes: nuclide concentrations only (with 
    input-related columns deleted) + 1 dataframe for each label column.

    Parameters
    ----------
    dfXY : dataframe with nuclide concentraations and 3 labels: reactor type, 
           enrichment, and burnup

    Returns
    -------
    dfX : dataframe with only nuclide concentrations for each instance
    r_dfY : dataframe with reactor type for each instance
    e_dfY : dataframe with fuel enrichment for each instance
    b_dfY : dataframe with fuel burnup for each instance

    """

    x = len(dfXY.columns)-3
    dfX = dfXY.iloc[:, 0:x]
    r_dfY = dfXY.iloc[:, x]
    e_dfY = dfXY.iloc[:, x+1]
    b_dfY = dfXY.iloc[:, x+2]
    return dfX, r_dfY, e_dfY, b_dfY

def main():
    """
    Takes all origen files and compiles them into the appropriate dataframes for 
    training and testing sets. Then splits those dataframes into the appropriate 
    X and Ys for prediction of reactor type, fuel enrichment, and burnup. 

    The training set is varied by number of features included in trainX to
    create a learning curve.

    """
    
    pkl_train = 'trainXY_2nov.pkl'
    pkl_test = 'testXY_2nov.pkl'
    print("Default scikit validation curve script is running...\n")
    #print("Did you check your training and testing data paths?\n")    
    # Training Datasets
    #trainpath = "../origen/origen-data/training/9may2017/csv/"
    #trainpath = "../origen-data/training/2nov2017/csv/"
    #trainpath = "../origen/origen-data/training/2nov2017/csv/"
    #train_files = glob.glob(os.path.join(trainpath, "*.csv"))
    #trainXY = dataframeXY(train_files, train_label)
    #trainXY.reset_index(inplace = True)
    #pickle.dump(trainXY, open(pkl_train, 'wb'))
    
    # Get set of top 200 nucs from training set
    # The filter_nuc func repeats stuff from top_nucs but it is needed because
    # the nuc_set needs to be determined from the training set for the test set
    # and the training set is filtered within each loop
    trainXY = pd.read_pickle(pkl_train)
    top_n = 200
    nuc_set = top_nucs(trainXY, top_n)
    trainX, trainYr, trainYe, trainYb = splitXY(trainXY)
    trainX = filter_nucs(trainX, nuc_set, top_n)
    trainX = scale(trainX)
    train_set = LearnSet(nuc_concs = trainX, burnup = trainYb)
    
    # Testing Dataset (for now)
    #testpath = "../origen/origen-data/testing/10may2017_2/csv/"
    #testpath = "../origen-data/testing/2nov2017/csv/"
    #testpath = "../origen/origen-data/testing/2nov2017/csv/"
    #test_files = glob.glob(os.path.join(testpath, "*.csv"))
    #testXY = dataframeXY(test_files, test_label)
    #testXY.reset_index(inplace = True)
    #pickle.dump(testXY, open(pkl_test, 'wb'))
    testXY = pd.read_pickle(pkl_test)
    testX, testYr, testYe, testYb = splitXY(testXY)
    testX = filter_nucs(testX, nuc_set, top_n)
    testX = scale(testX)
    test_set = LearnSet(nuc_concs = testX, burnup = testYb)
    
    #random_error(train_set, test_set)
    auto_train_and_predict(train_set) 
    manual_train_and_predict(train_set, test_set) 
    
    print("All csv files are saved in this directory!\n")

    return

if __name__ == "__main__":
    main()
