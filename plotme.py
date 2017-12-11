#! /usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import plotly as ply
import os

plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams['figure.figsize'] = ()

# Custom Color Blind Palette
DBROWN = '#8c510a'
MBROWN = '#d8b365'
LBROWN = '#f6e8c3'
DTEAL = '#01665e'
MTEAL = '#5ab4ac'
LTEAL = '#c7eae5'
DPURPLE = '#762a83'
MPURPLE = '#af8dc3'
LPURPLE = '#e7d4e8'

# Where the data are stored:
rpath = './results/11dec_knn/fissact/'
preds = ('reactor', 'cooling', 'enrichment', 'burnup')
src = ('_nucs', '_gammas')
###########################################
######### These two are hardcoded #########
subset = '_fissact'
tsubset = 'Fission Products and Actinides'
###########################################
print('You are plotting the {} subset'.format(subset), flush=True)
# Other useful lists for plotting
colors = (DBROWN, MBROWN, DTEAL, MTEAL)
labels = ('CVL1', 'TrainL1', 'CVL2', 'TrainL2')
quality = ('Nuclide Concentrations', 'Gamma spectra')
titles = ('Reactor Type', 'Cooling Time [days?]', 
          'Enrichment [%U235]', 'Burnup [MWd/MTU]'
          )

# Learning Curve Data
for s in src:
    i = src.index(s)
    for p in preds:
        j = preds.index(p)
        fig = plt.figure()  
        lcsv = 'lc_' + p + subset + s + '.csv'
        ldatapath = os.path.join(rpath, lcsv)
        ldata = pd.read_csv(ldatapath)
        X = ldata.iloc[:, 0]
        plt.xlabel('Training set size (m)')
        # y labels
        if j == 0:
            plt.ylabel('Accuracy Score')
        elif j == 3:
            # Make sure burnup is in M not 1e7
            ldata.iloc[:, 1:] = ldata.iloc[:, 1:].mul(10**(-6))
            plt.ylabel('Negative Mean-squared Error')
        else:
            plt.ylabel('Negative Mean-squared Error')
        for c, column in enumerate(labels):
            Y = ldata.loc[:, column]
            plt.plot(X, Y, label=column, color=colors[c])
        leg=plt.legend(loc='best', fancybox=True)
        # Very descriptive title for now
        plt_title = 'Learning Curve:\n' + titles[j] + ' Predictions \n from ' + \
                     tsubset + '\n of ' + quality[i]
        plt.title(plt_title, fontstyle='italic')
        # Save figure as PNG
        filename = 'lc_' + p + subset + s + '.png'
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)

        
## Validation Curve Data
for s in src:
    i = src.index(s)
    for p in preds:
        j = preds.index(p)
        fig = plt.figure()  
        vcsv = 'vc_' + p + subset + s + '.csv'
        vdatapath = os.path.join(rpath, vcsv)
        vdata = pd.read_csv(vdatapath)
        X = vdata.iloc[:, 0]
        plt.xlabel('Neighborhood Size (k)')
        # y labels
        if j == 0:
            plt.ylabel('Accuracy Score')
        elif j == 3:
            # Make sure burnup is in M not 1e7
            vdata.iloc[:, 1:] = vdata.iloc[:, 1:].mul(10**(-6))
            plt.ylabel('Negative Mean-squared Error')
        else:
            plt.ylabel('Negative Mean-squared Error')
        for c, column in enumerate(labels):
            Y = vdata.loc[:, column]
            plt.plot(X, Y, label=column, color=colors[c])
        plt.gca().invert_xaxis()
        leg=plt.legend(loc='best', fancybox=True)
        # Very descriptive title for now
        plt_title = 'Validation Curve:\n' + titles[j] + ' Predictions \n from ' + \
                     tsubset + '\n of ' + quality[i]
        plt.title(plt_title, fontstyle='italic')
        # Save figure as PNG
        filename = 'vc_' + p + subset + s + '.png'
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
