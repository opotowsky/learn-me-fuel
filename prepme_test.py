#! /usr/bin/env python3

import testing_set_1 as ts
#import testing_set_2 as ts
import pickle
import numpy as np
import pandas as pd
import glob
import csv
import os

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
    #print(data.shape)
    #print(coolings)
    #print(burnups)
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
        data = format_ndf(training_set['filename'])
        labeled = label_data(training_set, data)
        labeled.drop_duplicates(keep='last', inplace=True)
        all_data.append(labeled)
    dfXY = pd.concat(all_data, sort=True)
    dfXY.fillna(value=0, inplace=True)
    return dfXY

def create_train_labels():
    """
    Creates a list of dictionaries containing the entire training set from the 
    imported training set file

    Returns
    -------
    train_set : list of dictionaries, each of which contains the simulation 
                subsets by ORIGEN rxtr

    """
    data_set = []
    for rxtr_data in ts.testing_set:
        for o_rxtr in rxtr_data['rxtrs']:
            for enrich in rxtr_data['enrich']:
                data_set.append( {'ReactorType' : rxtr_data['type'],
                                  'OrigenReactor' : o_rxtr,
                                  'Enrichment' : enrich,
                                  'Burnups' : rxtr_data['burnup'],
                                  'CoolingInts' : rxtr_data['cooling_intervals'] } )
    return data_set

def main():
    """
    Takes all origen files in the hard-coded datapath and compiles them into 
    the appropriate dataframe for the training set. Saves the training set as
    a pickle file.

    """
    # Always check this data path
    origen_dir = '../origen/origen-data/'
    data_dir = '2jul2018_testset1'
    datapath = origen_dir + data_dir + '/' 
    print('Is {} the correct testing set directory?\n'.format(datapath), flush=True)
    # Grab data set
    data_set = create_train_labels()
    # Make pkl files according to nuc subset and measurement source
    subset = ['_fiss', '_act', '_fissact']#, '_all']
    info_src = ['_nucs',]# '_gammas']
    for nucs_tracked in subset:
        for src in info_src:
            files = {}
            i = 1
            for sim in data_set:
                o_rxtr = sim['OrigenReactor']
                enrich = sim['Enrichment']
                rxtrpath = datapath + o_rxtr + "/"
                csvfile = o_rxtr + '_' + str(i)  + "_enr" + str(enrich) + '_' + str(i) + nucs_tracked + src + ".csv"
                filepath = os.path.join(rxtrpath, csvfile)
                sim['filename'] = filepath
                i = i + 1
            dataXY = dataframeXY(data_set, src)
            pkl_name = data_dir + src + nucs_tracked + '_not-scaled.pkl'
            pickle.dump(dataXY, open(pkl_name, 'wb'), protocol=2)
    return

if __name__ == "__main__":
    main()
