#! /usr/bin/env python

from __future__ import print_function
from preds import train_and_predict
from training_set import *
import numpy as np
import pandas as pd
import glob
import os

class LearnSet(object):
    """
    A set of parameters (i.e., features and labels) for a machine learning 
    algorithm, each in the format of a pandas dataframe
    """

    def __init__(self, nuc_concs, reactor, enrichment, burnup):
        self.nuc_concs = nuc_concs
        self.reactor = reactor
        self.enrichment = enrichment
        self.burnup = burnup

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
                 'CoolingInts': COOLING_INTERVALS
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
    Takes all origen files and compiles them into the appropriate dataframe for 
    the training set. Then splits the dataframe into the appropriate X and Ys 
    for prediction of reactor type, cooling time, fuel enrichment, and burnup. 

    """
    
    print("Did you check your training and testing data paths?\n")    
    # Training Datasets
    trainpath = "../origen-data/14nov2017/"
    for o_rxtr in O_RXTRS:
        train_files = glob.glob(os.path.join(trainpath, "*.csv"))
    trainXY = dataframeXY(train_files, train_label)
    trainXY.reset_index(inplace=True)
    
    # formulate filtered training and testing sets
    trainX, trainYr, trainYe, trainYb = splitXY(trainXY)
    trainX = filter_nucs(trainX, nuc_set, top_n)
    trainX = trainX.astype(float)
    train_set = LearnSet(nuc_concs = trainX, reactor = trainYr, 
                         enrichment = trainYe, burnup = trainYb)    
    # Predict!
    train_and_predict(train_set, test_set)

    print("All csv files are saved in this directory!\n")

    return

if __name__ == "__main__":
    main()
