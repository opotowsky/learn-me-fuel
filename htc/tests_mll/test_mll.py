#! /usr/bin/env python3

from mll_calc.mll_calc import *
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def dfXY():
    unc = 1
    lbls = ['label']
    ll_name = 'LogLikelihood_' + str(unc)
    # Don't need 2nd feature for testing, yet?
    #XY = pd.DataFrame({'feature1' : [0, 1, 3], 
    #                   'feature2' : [1, 1, 1],
    #                   'label' : ['X', 'Y', 'Z']},
    #                   index = [0, 1, 2])
    XY = pd.DataFrame({'feature' : [1, 2, 3], 
                       'label' : ['X', 'Y', 'Z']},
                       index = [0, 1, 2])
    return XY, unc, lbls, ll_name

def ll_calc(x, std):
    # where x = y_sim - y_mes
    # where std = unc * y_sim
    ll = -0.5 * ((x / std)**2 + np.log(2 * np.pi) + 2 * np.log(std))
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

def test_get_pred_idx0(dfXY):
    sim_idx = 0
    XY, unc, lbls, ll_name = dfXY
    test_sample = XY.loc[sim_idx].drop(lbls)
    XY.drop(sim_idx, inplace=True)
    ll_exp = [ll_calc(1, 2)]
    exp_1 = pd.DataFrame({'pred_label' : ['Y'],
                          ll_name : ll_exp},
                          index = [1])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = get_pred(XY, test_sample, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

def test_get_pred_idx1(dfXY):
    sim_idx = 1
    XY, unc, lbls, ll_name = dfXY
    test_sample = XY.loc[sim_idx].drop(lbls)
    XY.drop(sim_idx, inplace=True)
    ll_exp = [ll_calc(1, 1)]
    exp_1 = pd.DataFrame({'pred_label' : ['X'],
                          ll_name : ll_exp},
                          index = [0])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = get_pred(XY, test_sample, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

def test_get_pred_idx2(dfXY):
    sim_idx = 2
    XY, unc, lbls, ll_name = dfXY
    test_sample = XY.loc[sim_idx].drop(lbls)
    XY.drop(sim_idx, inplace=True)
    ll_exp = [ll_calc(1, 2)]
    exp_1 = pd.DataFrame({'pred_label' : ['Y'],
                          ll_name : ll_exp},
                          index = [1])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = get_pred(XY, test_sample, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

def test_mll_testset_XY(dfXY):
    XY, unc, lbls, ll_name = dfXY
    test = XY.copy()
    ll_exp = [ll_calc(1, 2), ll_calc(1, 1), ll_calc(1, 2)]
    exp_1 = pd.DataFrame({'sim_idx' : [0, 1, 2],
                          'label' : ['X', 'Y', 'Z'],
                          'pred_idx' : [1, 0, 1],
                          'pred_label' : ['Y', 'X', 'Y'],
                          ll_name : ll_exp}, 
                          index = [0, 1, 2])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = mll_testset(XY, test, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

def test_mll_testset_ext(dfXY):
    XY, unc, lbls, ll_name = dfXY
    test = pd.DataFrame({'feature' : [4], 
                       'label' : ['W']},
                       index = ['A'])
    ll_exp = [ll_calc(1, 3)]
    exp_1 = pd.DataFrame({'sim_idx' : ['A'],
                          'label' : ['W'],
                          'pred_idx' : [2],
                          'pred_label' : ['Z'],
                          ll_name : ll_exp}, 
                          index = [0])
    exp_2 = ['pred_label', ll_name]
    obs_1, obs_2 = mll_testset(XY, test, unc, lbls)
    assert obs_1.equals(exp_1)
    assert obs_2 == exp_2

# def test_mll_testset_drop_replace():
# need to test that the db is in fact stating the same (not slowly getting deleted)
