#! /usr/bin/env python

from __future__ import print_function
from preds import reactor, enrichment, burnup
#from cv_preds import reactor, enrichment, burnup
import numpy as np
import pandas as pd
import glob
import os

class LearnSet(object):
    """
    A set of parameters (i.e., features and labels) for a machine learning 
    algorithm.
    """

    def __init__(self, nuc_concs, reactor, enrichment, burnup):
        self.nuc_concs = nuc_concs
        self.reactor = reactor
        self.enrichment = enrichment
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
    return dfXY

def splitXY(dfXY):
    """
    Takes a dataframe with all X (features) and Y (labels) information and 
    produces four different dataframes: nuclide concentrations only (with 
    input-related columns deleted) + 1 dataframe for each label column.

    Parameters
    ----------
    dfXY : dataframe with several input-related columns, nuclide concentraations, 
           and 3 labels: reactor type, enrichment, and burnup

    Returns
    -------
    dfX : dataframe with only nuclide concentrations for each instance
    r_dfY : dataframe with reactor type for each instance
    e_dfY : dataframe with fuel enrichment for each instance
    b_dfY : dataframe with fuel burnup for each instance

    """

    x = len(dfXY.columns)-3
    y = x
    # Need better way to know when the nuclide columns start (6 for now)
    # Prob will just search for column idx that starts with str(1)?
    dfX = dfXY.iloc[:, 6:x]
    # Best place to filter for top 200 nuclides is here 
    # (but spent 6 hours trying to figure out and failed)
    r_dfY = dfXY.iloc[:, y]
    e_dfY = dfXY.iloc[:, y+1]
    b_dfY = dfXY.iloc[:, y+2]
    return dfX, r_dfY, e_dfY, b_dfY

def random_error(percent_err, df):
    """
    Given a dataframe of nuclide vectors, add error to each element in each 
    nuclide vector that has a random value within the range [1-err, 1+err]

    Parameters
    ----------
    percent_err : a float indicating the maximum error that can be added to the nuclide 
                  vectors
    df : dataframe of only nuclide concentrations

    Returns
    -------
    df_err : dataframe with nuclide concentrations altered by some error

    """
    x = len(df)
    y = len(df.columns)
    err = percent_err / 100.0
    low = 1 - err
    high = 1 + err
    errs = np.random.uniform(low, high, (x, y))
    df_err = df * errs
    
    return df_err

def main():
    """
    Takes all origen files and compiles them into the appropriate dataframes for 
    training and testing sets. Then splits those dataframes into the appropriate 
    X and Ys for prediction of reactor type, fuel enrichment, and burnup.
    """
    
    #print("Did you check your training and testing data paths?\n")
    
    # Training Datasets
    trainpath = "../origen/origen-data/training/9may2017/csv/"
    train_files = glob.glob(os.path.join(trainpath, "*.csv"))
    trainXY = dataframeXY(train_files, train_label)
    trainXY.reset_index(inplace=True)
    trainX, r, e, b = splitXY(trainXY)
    train_set = LearnSet(nuc_concs = trainX, reactor = r, enrichment = e, burnup = b)
    
    # Testing Dataset (for now)
    testpath = "../origen/origen-data/testing/10may2017_2/csv/"
    test_files = glob.glob(os.path.join(testpath, "*.csv"))
    testXY = dataframeXY(test_files, test_label)
    testXY.reset_index(inplace=True)
    testX, rr, ee, bb = splitXY(testXY)
    
    # Add random errors of varying percents to nuclide vectors in the test set 
    # to mimic measurement error
    percent_err = np.arange(0.0, 1.25, 0.25)
    reactor_acc = []
    enrichment_err = []
    burnup_err = []
    for err in percent_err:
        if err == 0.0:
            test_set = LearnSet(nuc_concs = testX, reactor = rr, enrichment = ee, burnup = bb)
        else:
            testX_err = random_error(err, testX)
            test_set = LearnSet(nuc_concs = testX_err, reactor = rr, enrichment = ee, burnup = bb)
        # Predict!
        # l1 nn, l2 nn, ridge for each
        # reactor type is accuracy, e and b are RMSE
        rp = reactor(train_set, test_set)
        ep = enrichment(train_set, test_set)
        bp = burnup(train_set, test_set)
        reactor_acc.append(rp)
        enrichment_err.append(ep)
        burnup_err.append(bp)
    
    # Save results
    cols = ['L1NN', 'L2NN', 'RIDGE']
    pd.DataFrame(reactor_acc, columns=cols, index=percent_err).to_csv('reactor.csv')
    pd.DataFrame(enrichment_err, columns=cols, index=percent_err).to_csv('enrichment.csv')
    pd.DataFrame(burnup_err, columns=cols, index=percent_err).to_csv('burnup.csv')

    return

if __name__ == "__main__":
    main()
