#! /usr/bin/env python

from training_set import *
import pickle
import numpy as np
import pandas as pd
import glob
import csv
import os

def format_gdf(filename):
    """
    This takes a csv file and reads the data in as a dataframe.
    There are different requirements for the gamma file bc opus gives
    stupid output - so can't use pandas functionality

    Parameters
    ----------
    filename : str of simulation output in a csv file

    Returns
    -------
    data : pandas dataframe containing csv entries

    """
    time_idx = []
    spectrum = []
    spectra = []
    gamma_bins = ()
    with open(filename) as f:
        gamma = csv.reader(f, delimiter=',')
        i = 1
        for row in gamma:
            if len(row) > 0:
                if i < 6:
                    pass
                elif i == 6:
                    time_idx.append(row[0])
                elif row[1]=='days':
                    spectra.append(spectrum)
                    time_idx.append(row[0])
                    spectrum = []
                else:
                    if i in range(7, 209):
                        if (i > 7 and gamma_bins[-1]==row[0]):
                            row[0] = row[0] + '.1'
                        gamma_bins = gamma_bins + (row[0],)    
                    spectrum.append(row[1])
                i = i + 1
        spectra.append(spectrum)
    data = pd.DataFrame(spectra, index=time_idx, columns=gamma_bins)
    return data

def format_ndf(filename):
    """
    This takes a csv file and reads the data in as a dataframe.

    Parameters
    ----------
    filename : str of simulation output in a csv file

    Returns
    -------
    data : pandas dataframe containing csv entries

    """
    
    data = pd.read_csv(filename, header=5, index_col=0).T
    data.drop('subtotal', axis=1, inplace=True)
    return data

def label_data(labels, data):
    """
    Takes the labels for and a dataframe of the simulation results; 
    adds these labels as additional columns to the dataframe.

    Parameters
    ----------
    labels : dict representing the labels for a simulation
    data : dataframe of simulation results

    Returns
    -------
    data : dataframe of simulation results + label entries in columns

    """
    
    col = len(data.columns)
    burnups, coolings = loop_labels(labels['Burnup'], labels['CoolingInts'])
    # inserting 4 labels into columns
    data.insert(loc = col, column = 'ReactorType', value = labels['ReactorType'])
    data.insert(loc = col+1, column = 'CoolingTime', value = coolings)
    data.insert(loc = col+2, column = 'Enrichment', value = labels['Enrichment'])
    data.insert(loc = col+3, column = 'Burnup', value = burnups)
    return data

def loop_labels(burnup, cooling):
    """
    Takes the burnups and cooling time for each case within the simulation and
    creates a list of the burnup of the irradiated and cooled/ decayed fuels;
    returns a list to be added as the burnup label to the main dataframe.

    Parameters
    ----------
    burnup : list of the steps of burnup from the simulation parameters
    cooling : list of the cooling intervals from the simulation parameters

    Returns
    -------
    burnup_lbl : list of burnups to be applied as a label for a given simulation
    cooling_lbl : list of cooling times to be applied as a label for a given simulation

    """
    
    steps_per_case = len(COOLING_INTERVALS)
    burnup_lbl = [0,]
    cooling_lbl = [0,]
    for case in range(0, len(burnup)):
        if case == 0:
            pass
        else:
            # corresponds to previous material logging step
            burnup_lbl.append(burnup[case-1])
            cooling_lbl.append(0)
        # corresponds to irradiation step
        burnup_lbl.append(burnup[case])
        cooling_lbl.append(0)
        for step in range(0, steps_per_case):
            # corresponds to 5 cooling times
            burnup_lbl.append(burnup[case])
            cooling_lbl.append(COOLING_INTERVALS[step])
    return burnup_lbl, cooling_lbl

def dataframeXY(all_files, info):
    """" 
    Takes list of all files in a directory (and rxtr-labeled subdirectories) 
    and produces a dataframe that has both the data features (X) and labeled 
    data (Y).

    Parameters
    ----------
    all_files : list of str holding all simulation file names in a directory
    info : string indicating the information source of the training data

    Returns
    -------
    dfXY : dataframe that has all features and labels for all simulations in a 
           directory

    """

    all_data = []
    for f in all_files:
        idx = all_files.index(f)
        if info == '_gammas':
            data = format_gdf(f)
        else:
            data = format_ndf(f)
        labels = {'ReactorType': TRAIN_LABELS['ReactorType'][idx],
                  #'OrigenReactor': TRAIN_LABELS['OrigenReactor'][idx],
                  'Enrichment': TRAIN_LABELS['Enrichment'][idx], 
                  'Burnup': TRAIN_LABELS['Burnup'][idx], 
                  'CoolingInts': COOLING_INTERVALS
                  }
        labeled = label_data(labels, data)
        labeled.drop_duplicates(keep='last', inplace=True)
        all_data.append(labeled)
    dfXY = pd.concat(all_data)
    dfXY.fillna(value=0, inplace=True)
    return dfXY


def main():
    """
    Takes all origen files in the hard-coded datapath and compiles them into 
    the appropriate dataframe for the training set. Saves the training set as
    a pickle file.

    """
    
    print("Did you check your training data path?\n", flush=True)
    info_src = ['_nucs', '_gammas']
    #datapath = "../origen/origen-data/8dec2017/"
    datapath = "../origen-data/8dec2017/"
    subset = ['_fiss', '_act', '_fissact']
    for nucs_tracked in subset:
        for src in info_src:
            train_files = []
            for i in range(0, len(O_RXTRS)):
                o_rxtr = O_RXTRS[i]
                for j in range(0, len(ENRICH[i])):
                    enrich = ENRICH[i][j]
                    rxtrpath = datapath + o_rxtr + "/"
                    csvfile = o_rxtr + "_enr" + str(enrich) + nucs_tracked + src + ".csv"
                    trainpath = os.path.join(rxtrpath, csvfile)
                    train_files.append(trainpath)
            trainXY = dataframeXY(train_files, src)
            pkl_name = 'not-scaled_trainset' + src + nucs_tracked + '_8dec.pkl'
            pickle.dump(trainXY, open(pkl_name, 'wb'))
    return

if __name__ == "__main__":
    main()
