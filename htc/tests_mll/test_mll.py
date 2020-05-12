#! /usr/bin/env python3

from mll_calc.mll_calc import *

import numpy as np
import pandas as pd

def ll_calc(x):
    # where x = y_sim - y_mes
    ll = -0.5 * (x**2 + np.log(2 * np.pi))
    return ll

# dummy test (don't need to test single line scipy func, I think)
def test_like_calc():
    y_sim = pd.Series([1, 1, 1, 1])
    y_mes = pd.Series([1, 1, 1, 1])
    std = pd.Series([1, 1, 1, 1])
    exp = (1 / np.sqrt(2 * np.pi)) ** len(y_sim)
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

def test_get_pred():
    XY = pd.DataFrame({'feature' : [1, 3], 
                       'label' : ['Y', 'Z']},
                       index = [1, 2])
    test_sample = pd.Series([0], index=['feature'])
    unc = 1
    lbls = ['label']
    ll_name = 'LogLikelihood_' + str(unc)
    exp_1 = pd.DataFrame({'pred_label' : ['Y'],
                          ll_name : [ll_calc(1)]},
                          index = [1])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = get_pred(XY, test_sample, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

def test_mll_testset_XY():
    XY = pd.DataFrame({'feature' : [1., 2., 4.], 
                       'label' : ['X', 'Y', 'Z']},
                       index = [0, 1, 2])
    test = XY.copy()
    unc = 1
    lbls = ['label']
    ll_name = 'LogLikelihood_' + str(unc)
    exp_1 = pd.DataFrame({'sim_idx' : [0, 1, 2],
                          'label' : ['X', 'Y', 'Z'],
                          'pred_idx' : [1, 0, 1],
                          'pred_label' : ['Y', 'X', 'Y'],
                          ll_name : [ll_calc(1), ll_calc(1), ll_calc(2)]}, 
                          index = [0, 1, 2])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = mll_testset(XY, test, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

#def test_mll_testset_ext():
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
#    obs_1, obs_2 = mll_testset(XY, test, unc, lbls)
#    assert obs_1.equals(exp_1)
#    assert obs_2 == exp_2

# def test_mll_testset_drop_replace():
# need to test that the db is in fact stating the same (not slowly getting deleted)
