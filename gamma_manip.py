#! /usr/bin/env python

from training_set import *
from preds import train_and_predict
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import glob
import csv
import os

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
    data.drop_duplicates(keep='last', inplace=True)    
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
    data.insert(loc = col+1, column = 'Enrichment', value = labels['Enrichment'])
    data.insert(loc = col+2, column = 'Burnup', value = burnups)
    data.insert(loc = col+3, column = 'CoolingTime', value = coolings)
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
    
    steps_per_case = len(COOLING_INTERVALS) + 1
    burnup_lbl = [0, ]
    cooling_lbl = [0, ]
    for case in range(0, len(burnup)):
        for step in range(0, steps_per_case):
            if (step == 0):
                burnup_lbl.append(burnup[case])
                cooling_lbl.append(0)
            else:
                burnup_lbl.append(burnup[case])
                cooling_lbl.append(COOLING_INTERVALS[step-1])
    return burnup_lbl, cooling_lbl

def dataframeXY(all_files):
    """" 
    Takes list of all files in a directory (and rxtr-labeled subdirectories) 
    and produces a dataframe that has both the data features (X) and labeled 
    data (Y).

    Parameters
    ----------
    all_files : list of str holding all simulation file names in a directory

    Returns
    -------
    dfXY : dataframe that has all features and labels for all simulations in a 
           directory

    """

    all_data = []
    for f in all_files:
        idx = all_files.index(f)
        data = format_df(f)
        labels = {'ReactorType': TRAIN_LABELS['ReactorType'][idx],
                  #'OrigenReactor': TRAIN_LABELS['OrigenReactor'][idx],
                  'Enrichment': TRAIN_LABELS['Enrichment'][idx], 
                  'Burnup': TRAIN_LABELS['Burnup'][idx], 
                  'CoolingInts': COOLING_INTERVALS
                  }
        labeled = label_data(labels, data)
        all_data.append(labeled)
    dfXY = pd.concat(all_data)
    dfXY.fillna(value=0, inplace=True)
    return dfXY

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

    """

    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup']
    dfX = dfXY.drop(lbls, axis=1)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY

def main():
    """
    Takes all origen files and compiles them into the appropriate dataframe for 
    the training set. Then splits the dataframe into the appropriate X and Ys 
    for prediction of reactor type, cooling time, fuel enrichment, and burnup. 

    """
    
    print("Did you check your training and testing data paths?\n")    
    train_files = []
    #datapath = "../origen/origen-data/30nov2017_actinides/"
    datapath = "../origen-data/30nov2017_actinides/"
    for i in range(0, len(O_RXTRS)):
        o_rxtr = O_RXTRS[i]
        for j in range(0, len(ENRICH[i])):
            enrich = ENRICH[i][j]
            rxtrpath = datapath + o_rxtr + "/"
            csv = o_rxtr + "_enr" + str(enrich) + "_gammas.csv"
            trainpath = os.path.join(rxtrpath, csv)
            train_files.append(trainpath)
    
    trainXY = dataframeXY(train_files)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX, with_mean=False)
    train_and_predict(trainX, rY, cY, eY, bY)

    return

if __name__ == "__main__":
    main()
