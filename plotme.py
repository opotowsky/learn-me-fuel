#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import plotly as ply
import os

plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams[''nes.linewidth] = 2.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
#plt.rcParams['figure.figsize'] = ()

# Custom Color Blind Palette
BROWN = '#8c510a'
MBROWN = '#d8b365'
LBROWN = '#f6e8c3'
DTEAL = '#01665e'
MTEAL = '#5ab4ac'
LTEAL = '#c7eae5'
DPURPLE = '#762a83'
MPURPLE = '#af8dc3'
LPURPLE = '#e7d4e8'

# Where the data are stored:
preds = ['reactor', 'enrichment', 'cooling', 'burnup']
src = ['_nucs', '_gammas']
###########################################
######### These two are hardcoded #########
rpath = './results/8dec_knn/fissionprods/'#
subset = '_fiss'                          #
tsubset = 'Fission Products'              #
###########################################
print('You are plotting the {} subset'.format(subset), flush=True)
# Other useful lists for plotting
colors = [DBROWN, MBROWN, DTEAL, MTEAL]
labels = ['CVL1', 'TrainL1', 'CVL2', 'TrainL2']     
quality = ['Nuclide Concentrations', 'Gamma spectra']
titles = ['Reactor Type', 'Enrichment [%U235]', 
          'Burnup [MWd/MTU]', 'Cooling Time [min?]'
          ]

# Learning Curve Data
for i, s in enumerate(src):
    for j, p in enumerate(preds):
        axl = plt.subplot()  
        lcsv = 'lc_' + p + subset + s + '.csv'
        ldatapath = os.path.join(rpath, lcsv)
        ldata = pd.read_csv(ldatapath)
        plt.xlabel('Training set size (m)')
        # y labels
        if j == 0:
            plt.ylabel('Accuracy Score')
        else:
            plt.ylabel('Negative Mean-squared Error')
        for lrank, column in enumerate(ldata.columns):  
            ldata[column].plot(ldata.X, ldata.Y, ax=axl, color=colors[lrank])
        # Customize title location
        plt_title = 'Learning Curve: ' + titles[j] + ' Predictions from ' + 
                    tsubset + ' of ' + quality[i]
        plt.text(0.5*(left+right), top, plt_title, fontsize=16, ha='center', va='bottom')
        # Save figure as PNG
        filename = 'lc_' + p + subset + s + '.png'
        plt.savefig(filename, bbox_inches="tight")

        
# Validation Curve Data
for i, s in enumerate(src):
    for j, p in enumerate(preds):
        fig = plt.figure()  
        vcsv = 'vc_' + p + subset + s + '.csv'
        vdatapath = os.path.join(rpath, vcsv)
        vdata = pd.read_csv(vdatapath)
        plt.xlabel('Neighborhood size (k)')
        # y labels
        if j == 0:
            plt.ylabel('Accuracy Score')
        else:
            ply.ylabel('Negative Mean-squared Error')
