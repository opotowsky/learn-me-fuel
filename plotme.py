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
plt.rcParams['figure.titlesize'] = 14
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
rpath = './results/8dec_knn/fissionprods/'
preds = 'reactor', 'cooling', 'enrichment', 'burnup')
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
          'Enrichment [%U235]', 'Burnup [?Wd/MTU]'
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
        else:
            plt.ylabel('Negative Mean-squared Error')
        for column in labels:
            Y = ldata.loc[:, column]
            plt.plot(X, Y, label=column, color=colors.index(column))
        leg=plt.legend(loc='best', fancybox=True)
        # Customize title location
        plt_title = 'Learning Curve: ' + titles[j] + ' Predictions from ' + \
                     tsubset + ' of ' + quality[i]
        plt.title(plt_title, fontstyle='italic')
        # Save figure as PNG
        filename = 'lc_' + p + subset + s + '.png'
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)

        
## Validation Curve Data
#for i, s in enumerate(src):
#    for j, p in enumerate(preds):
#        fig = plt.figure()  
#        vcsv = 'vc_' + p + subset + s + '.csv'
#        vdatapath = os.path.join(rpath, vcsv)
#        vdata = pd.read_csv(vdatapth)
#        plt.xlabel('Neighborhood size (k)')
#        # y labels
#        if j == 0:
#            plt.ylabel('Accuracy Score')
#        else:
#            ply.ylabel('Negative Mean-squared Error')
