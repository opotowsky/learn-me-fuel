
# coding: utf-8

# In[1]:

from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn import metrics
import numpy as np
import pandas as pd
import glob
import os


# In[2]:

# Info for labeling data
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


# In[3]:

# For labeling the actual values in the testing set
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


# In[4]:

def format_df(filename):
    # This is an instance of 
    data = pd.read_csv(filename).T
    data.columns = data.iloc[0]
    data.drop(data.index[0], inplace=True)
    return data

def get_labels(filename, rxtrs):
    tail, _ = os.path.splitext(os.path.basename(filename))
    i = rxtrs['OrigenReactor'].index(tail)
    rxtr_info = {'ReactorType': rxtrs['ReactorType'][i], 
                 'Enrichment': rxtrs['Enrichment'][i], 
                 'Burnup': rxtrs['Burnup'][i], 
                 'CoolingInts': rxtrs['CoolingInts'][i]
                 }
    return rxtr_info

def label_data(label, data):
    col = len(data.columns)
    data.insert(loc = col, column = 'ReactorType', value = label['ReactorType'])
    data.insert(loc = col+1, column = 'Enrichment', value = label['Enrichment'])
    burnup = burnup_label(data, label['Burnup'], label['CoolingInts'])
    data.insert(loc = col+2, column = 'Burnup', value = burnup)
    return data

def burnup_label(data, burn_steps, cooling_ints):
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
    all_data = []
    for f in all_files:
        data = format_df(f)
        labels = get_labels(f, rxtr_label)
        labeled = label_data(labels, data)
        all_data.append(labeled)
    dfXY = pd.concat(all_data)
    return dfXY

def top_nucs(df):
    # for each instance (row), keep only top 200 values, replace rest with 0 (scikit-learn won't accept NaN)
    top_n = 200
    ##################################################################
    # This gave seemingly uncorrelated indices -- needs exploration ##
    #top_cols = {}
    #for instance in df.itertuples():
    #    top_nucs = np.argpartition(instance[1:], -top_n)[-top_n:]
    #    top_cols[instance[0]] = top_nucs
    #print(top_cols[2000])
    #?????cond = [df.iloc[:, df.columns != col] for col in top_cols]
    #?????df.where(cond, other = np.nan, inplace=True)
    ##################################################################
    
    ########################################################################
    # This worked but made column lengths different between training sets ##
    #func = lambda x: x.sort_values(ascending=False)[:top_n]
    #top_cols = df.apply(func, axis=1)
    ########################################################################
    
    #######################################
    # This gave KeyError '      10001001'##
    #cols = list(df.T.columns.values)
    #top_cols = np.array([df.T[nuc].nlargest(top_n, cols).index.values for nuc, nums in df.T.iterrows()])
    #######################################
    
    #top_cols.fillna(value=0, inplace=True)
    #print(top_cols)
    return df

def splitXY(dfXY):
    x = len(dfXY.columns)-3
    y = x
    # Need better way to know when the nuclide columns start (6 for now)
    # Prob will just search for column idx that starts with str(1)?
    dfX = dfXY.iloc[:, 6:x]
    # Best place to filter for top 200 nuclides: 
    # (but spent 6 hours trying to figure out and failed)
    #dfX = top_nucs(dfX)
    r_dfY = dfXY.iloc[:, y]
    e_dfY = dfXY.iloc[:, y+1]
    b_dfY = dfXY.iloc[:, y+2]
    return dfX, r_dfY, e_dfY, b_dfY


# In[5]:

# Training Dataset
###############################################
## Still need to filter for top 200 nuclides ##
###############################################
trainpath = "../origen/origen-data/training/9may2017/csv/"
train_files = glob.glob(os.path.join(trainpath, "*.csv"))
trainXY = dataframeXY(train_files, train_label)
trainXY.reset_index(inplace=True)
trainX, r_trainY, e_trainY, b_trainY = splitXY(trainXY)
# Testing Dataset (for now)
testpath = "../origen/origen-data/testing/10may2017_2/csv/"
test_files = glob.glob(os.path.join(testpath, "*.csv"))
testXY = dataframeXY(test_files, test_label)
testXY.reset_index(inplace=True)
testX, r_testY, e_testY, b_testY = splitXY(testXY)


# In[6]:

# Reactor Type
# L1 norm is Manhattan Distance
# L2 norm is Euclidian Distance 
# Ridge Regression is Linear + L2 regularization
l1knc = KNeighborsClassifier(metric='l1', p=1)
l2knc = KNeighborsClassifier(metric='l2', p=2)
rc = RidgeClassifier()
l1knc.fit(trainX, r_trainY)
l2knc.fit(trainX, r_trainY)
rc.fit(trainX, r_trainY)
# Predictions
predict1 = l1knc.predict(testX)
predict2 = l2knc.predict(testX)
predict3 = rc.predict(testX)
expected = r_testY
print(metrics.classification_report(expected, predict1))
print(metrics.classification_report(expected, predict2))
print(metrics.classification_report(expected, predict3))


# In[7]:

# Enrichment
l1knr = KNeighborsRegressor(metric='l1', p=1)
l2knr = KNeighborsRegressor(metric='l2', p=2)
rr = Ridge()
l1knr.fit(trainX, e_trainY)
l2knr.fit(trainX, e_trainY)
rr.fit(trainX, e_trainY)
predict1 = l1knr.predict(testX)
predict2 = l2knr.predict(testX)
predict3 = rr.predict(testX)
expected = e_testY
print(metrics.mean_absolute_error(expected, predict1))
print(metrics.mean_absolute_error(expected, predict2))
print(metrics.mean_absolute_error(expected, predict3))


# In[8]:

# Burnup
bl1knr = KNeighborsRegressor(metric='l1', p=1)
bl2knr = KNeighborsRegressor(metric='l2', p=2)
brr = Ridge()
bl1knr.fit(trainX, b_trainY)
bl2knr.fit(trainX, b_trainY)
brr.fit(trainX, b_trainY)
predict1 = bl1knr.predict(testX)
predict2 = bl2knr.predict(testX)
predict3 = brr.predict(testX)
expected = b_testY
print(metrics.mean_absolute_error(expected, predict1))
print(metrics.mean_absolute_error(expected, predict2))
print(metrics.mean_absolute_error(expected, predict3))
print(metrics.r2_score(expected, predict1))
print(metrics.r2_score(expected, predict2))
print(metrics.r2_score(expected, predict3))


# In[ ]:



