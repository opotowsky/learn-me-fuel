import matplotlib.pyplot as plt
import pandas as pd
import plotly as ply
import os

# Much of this was from:
# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/

# Color Blind Palette
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
rpath = './results/7dec2017_knn_actinides/'
preds = ['reactor', 'enrichment', 'cooling', 'burnup']
src = ['_nucs', '_gammas']
# Other useful lists for plotting
quality = ['nuclide concentrations', 'gamma spectra']
titles = [': Reactor Type', ': Enrichment [%U235]', ': Burnup [MWd/MTU]', ': Cooling Time [min?]']
curve = ['LearningCurve', 'ValidationCurve']

# Read the data into a pandas DataFrame
for i, s in enumerate(src):
    for j, p in enumerate(preds):
        csv = p + s + '.csv'
        datapath = os.path.join(rpath, csv)
        data = pd.read_csv(datapath)
        for k, c in enumerate(curve):
            # Common sizes: (10, 7.5) and (12, 9)  
            plt.figure(figsize=(12, 9))
            ax = plt.subplot(111)  
            # Ticks only on bottom and left  
            ax.get_xaxis().tick_bottom()  
            ax.get_yaxis().tick_left()  
            # Plot range 
            #plt.ylim(0, 90)  
            #plt.xlim(1968, 2014)  
            # Large ticks   
            plt.yticks(fontsize=14)#range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)  
            plt.xticks(fontsize=14)  
            colors = [DBROWN, MBROWN, DTEAL, MTEAL]
            labels = ['CVL1', 'TrainL1', 'CVL2', 'TrainL2'] 
            
            # split data into 2 here
            # x labels
            if k == 0:
                data = data[]
                ax.xlabel('Neighborhood size (k)')
            else:
                data = data[]
                ax.xlabel('Training set size (m)')
            # y labels
            if j == 0:
                ax.ylabel('Accuracy Score')
            else:
                ax.ylabel('Negative Mean-squared Error')

            for lrank, column in enumerate(labels):  
                plt.plot(data.X.valyes,  
                         data.Y.values,  
                         lw=2.5, color=colors[lrank]
                         )

            # Customize title location
            plt_title = c + titles[j] + ' predictions from ' + quality[i]
            plt.text(0.5*(left+right), top, plt_title, fontsize=17, ha='center', va='bottom')

            # Save figure as PNG
            filename = p + '_' + c + '_' + s + '_actinides.png'
            plt.savefig(filename, bbox_inches="tight")

