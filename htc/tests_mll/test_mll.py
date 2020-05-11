#! /usr/bin/env python3

from mll_calc.mll_calc import *

import math
import numpy as np
import pandas as pd

# dummy test (don't need to test single line scipy func, I think)
def test_like_calc():
    y_sim = pd.Series([1, 1, 1, 1])
    y_mes = pd.Series([1, 1, 1, 1])
    std = pd.Series([1, 1, 1, 1])
    exp = (1 / math.sqrt(2 * math.pi)) ** len(y_sim)
    obs = like_calc(y_sim, y_mes, std)
    assert obs == exp

def test_ratios():
    ratio_list = ['A/B', 'B/A']
    labels = ['label']
    XY = pd.DataFrame({'A' : [1., 2., 3., np.nan], 
                       'B' : [1., 1., 0., 0],
                       'label' : [1, 1, 1, 1]})
    exp = pd.DataFrame({'A/B' : [1., 2., 0., 0.], 
                        'B/A' : [1., 0.5, 0, 0],
                        'label' : [1, 1, 1, 1]})
    obs = ratios(XY, ratio_list, labels)
    assert obs.equals(exp)

def test_testset_mll_XY():
    XY = pd.DataFrame({'A' : [0, 0, 0, 0], 
                       'B' : [0, 0, 0, 0],
                       'C' : [0, 0, 0, 0],
                       'label' : [1, 1, 1, 1]})
    test = XY.copy()
    unc = 1
    lbls = ['label']
    exp_1 = pd.DataFrame({'A' : [1., 2., 3., np.nan], 
                          'B' : [1., 1., 0., 0],
                          'label' : [1, 1, 1, 1]})
    exp_2 = ['Pred_label']
    obs_1, obs_2 = testset_mll(XY, test, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

#def test_testset_mll_ext():
#
#    XY = pd.DataFrame({'A' : [1., 2., 3., np.nan], 
#                       'B' : [1., 1., 0., 0],
#                       'label' : [1, 1, 1, 1]})
#    test = pd.DataFrame({'A' : [1., 2., 3., np.nan], 
#                         'B' : [1., 1., 0., 0],
#                         'label' : [1, 1, 1, 1]})
#    unc = 1
#    lbls = ['label']
#    exp_1 = pd.DataFrame({'A' : [1., 2., 3., np.nan], 
#                          'B' : [1., 1., 0., 0],
#                          'label' : [1, 1, 1, 1]})
#    exp_2 = ['Pred_label']
#    obs_1, obs_2 = testset_mll(XY, test, unc, lbls)
#    assert obs_1.equals(exp_1)
#    assert obs_2 == exp_2

# def test_testset_mll_drop_replace():
# need to test that the db is in fact stating the same (not slowly getting deleted)
