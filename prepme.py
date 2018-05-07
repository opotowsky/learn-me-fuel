#! /usr/bin/env python

import training_set as ts
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
    cooling_tmp = [0] + labels['CoolingInts'] + [0]
    burnups =  [ burnup for burnup in labels['Burnup'] for cooling in cooling_tmp ]
    coolings = cooling_tmp * len(labels['Burnup'])

    # the above process puts an extra entry on the end of each list
    burnups.pop()
    coolings.pop()
    
    # inserting 4 labels into columns
    data.insert(loc = col, column = 'ReactorType', value = labels['ReactorType'])
    data.insert(loc = col+1, column = 'CoolingTime', value = coolings )
    data.insert(loc = col+2, column = 'Enrichment', value = labels['Enrichment'])
    data.insert(loc = col+3, column = 'Burnup', value = burnups)
    # added the origen reactor for indepth purposes
    data.insert(loc = col+4, column = 'OrigenReactor', value = labels['OrigenReactor'])
    return data

def dataframeXY(train_labels, info):
    """" 
    Takes list of all files in a directory (and rxtr-labeled subdirectories) 
    and produces a dataframe that has both the data features (X) and labeled 
    data (Y).

    Parameters
    ----------
    train_labels : list of dicts holding training lables and filenames
    info : string indicating the information source of the training data

    Returns
    -------
    dfXY : dataframe that has all features and labels for all simulations in a 
           directory

    """

    all_data = []
    col_order = []
    for training_set in train_labels:
        if info == '_gammas':
            data = format_gdf(training_set['filename'])
        else:
            data = format_ndf(training_set['filename'])
        labeled = label_data(training_set, data)
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
    info_src = ['_nucs',] '_gammas']
    #datapath = "../origen/origen-data/8dec2017/"
    datapath = "../origen-data/8dec2017/"
    subset = ['_fissact',] '_act', '_fissact']
    for nucs_tracked in subset:
        for src in info_src:
            train_files = {}
            for training_set in ts.train_labels:
                o_rxtr = training_set[1]
                enrich = training_set[2]
                rxtrpath = datapath + o_rxtr + "/"
                csvfile = o_rxtr + "_enr" + str(enrich) + nucs_tracked + src + ".csv"
                trainpath = os.path.join(rxtrpath, csvfile)
                training_set['filename'] = trainpath
            trainXY = dataframeXY(ts.train_labels, src)
            pkl_name = 'not-scaled_trainset' + src + nucs_tracked + '_8dec.pkl'
            pickle.dump(trainXY, open(pkl_name, 'wb'), protocol=2)
    return

if __name__ == "__main__":
    main()
