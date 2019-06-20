#! /usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
import glob
import csv
import os

def format_gdf(filename):
    """
    This takes a filepath and reads the csv data in as a dataframe.
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
    This takes a filepath and reads the csv data in as a dataframe.

    Parameters
    ----------
    filename : str of simulation output in a csv file

    Returns
    -------
    data : pandas dataframe containing csv entries

    """
    
    data = pd.read_csv(filename, header=5, index_col=0).T
    if 'subtotal' in data.columns:
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
    cooling_tmp = [0] + list(labels['CoolingInts']) + [0]
    burnups =  [ burnup for burnup in labels['Burnups'] for cooling in cooling_tmp ]
    coolings = cooling_tmp * len(labels['Burnups'])

    # the above process puts an extra entry on the end of each list
    burnups.pop()
    coolings.pop()

    #still need a pre-0 to start
    burnups.insert(0, 0)
    coolings.insert(0, 0)

    # inserting 4 labels into columns
    data.insert(loc = col, column = 'ReactorType', value = labels['ReactorType'])
    data.insert(loc = col+1, column = 'CoolingTime', value = tuple(coolings))
    data.insert(loc = col+2, column = 'Enrichment', value = labels['Enrichment'])
    data.insert(loc = col+3, column = 'Burnup', value = tuple(burnups))
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
    dfXY = pd.concat(all_data, sort=True)
    dfXY.fillna(value=0, inplace=True)
    return dfXY

def main():
    """
    Takes all origen files in the hard-coded datapath and compiles them into
    the appropriate dataframe for a data set ready for pandas/scikit learn use.
    Saves the data set as a pickle file.

    """
    origen_dir = '../origen/origen-data/gendata/'
    data_dir = '22jul2018_trainset3' # '2jul2018_testset1' if using test set
    datapath = origen_dir + data_dir + '/' 
    testset = True if 'test' in datapath
    print('Is {} the correct data set directory?\n'.format(datapath), flush=True)
    # Grab data set labels
    if testset == True:
        print('Note from June 2019: Need to re-run generate_testingdata.py in origen-data project to get the .pkl file!', flush=True)
        pkl_labels = datapath + 'test_set.pkl'
    else:
        pkl_labels = datapath + 'varied_tset.pkl'
    data_set_labels = pickle.load(open(pkl_labels, 'rb'))
    # Make pkl files according to nuc subset and measurement source
    subset = ['_all', '_fiss', '_act', '_fissact']
    info_src = ['_nucs',]# '_gammas']
    for nucs_tracked in subset:
        for src in info_src:
            i = 1
            for sim in data_set_labels:
                o_rxtr = sim['OrigenReactor']
                enrich = sim['Enrichment']
                rxtrpath = datapath + o_rxtr + "/"
                if testset == True:
                    csvfile = o_rxtr + '_' + str(i)  + "_enr" + str(enrich) + '_' + str(i) + nucs_tracked + src + ".csv"
                else:
                    csvfile = o_rxtr + "_enr" + str(enrich) + nucs_tracked + src + ".csv"
                filepath = os.path.join(rxtrpath, csvfile)
                sim['filename'] = filepath
                i = i + 1
            dataXY = dataframeXY(data_set_labels, src)
            pkl_set = data_dir + src + nucs_tracked + '_not-scaled.pkl'
            pickle.dump(dataXY, open(pkl_set, 'wb'), protocol=2)
    return

if __name__ == "__main__":
    main()
