#! /usr/bin/env python3

from mll_calc.mll_calc import format_XY, ratios, get_pred, mll_testset, calc_errors, parse_args
import pickle
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def dfXY():
    unc = 1
    lbls = ['label']
    ll_name = 'LogLikelihood_' + str(unc)
    XY = pd.DataFrame({'feature' : [1, 2, 3], 
                       'label' : ['X', 'Y', 'Z']},
                       index = [0, 1, 2])
    return XY, unc, lbls, ll_name

def calc_ll_exp(x, std):
    # where x = y_sim - y_mes
    # where std = unc * y_sim
    ll = -0.5 * ((x / std)**2 + np.log(2 * np.pi) + 2 * np.log(std))
    return ll

def test_format_XY(tmpdir, dfXY):
    XY = pd.DataFrame({'feature' : [1, 2, 3],
                       'total' : [1, 2, 3],
                       'Burnup' : [1, 1, 0]},
                       index = [0, 1, 2])
    pkl_db = tmpdir.join('db.pkl')
    pickle.dump(XY, open(pkl_db, 'wb'))
    exp = pd.DataFrame({'feature' : [1, 2], 
                        'Burnup' : [1, 1]},
                        index = [0, 1])
    obs = format_XY(pkl_db)
    assert obs.equals(exp)

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

@pytest.mark.parametrize('sim_idx, exp',
                         [(0, pd.DataFrame({'pred_label' : ['Y'], 'LL' : [calc_ll_exp(1, 2)]}, index = [1])),
                          (1, pd.DataFrame({'pred_label' : ['X'], 'LL' : [calc_ll_exp(1, 1)]}, index = [0])),
                          (2, pd.DataFrame({'pred_label' : ['Y'], 'LL' : [calc_ll_exp(1, 2)]}, index = [1]))
                          ]
                         )
def test_get_pred(dfXY, sim_idx, exp):
    XY, unc, lbls, ll_name = dfXY
    test_sample = XY.loc[sim_idx].drop(lbls)
    XY.drop(sim_idx, inplace=True)
    #renaming LL col for now, until I understand parametrization with fixures
    exp.rename(columns={'LL': ll_name}, inplace=True)
    obs = get_pred(XY, test_sample, unc, lbls)
    assert obs.equals(exp)

def test_mll_testset_XY(dfXY):
    XY, unc, lbls, ll_name = dfXY
    test = XY.copy()
    ll_exp = [calc_ll_exp(1, 2), calc_ll_exp(1, 1), calc_ll_exp(1, 2)]
    exp = pd.DataFrame({'sim_idx' : [0, 1, 2],
                          'label' : ['X', 'Y', 'Z'],
                          'pred_idx' : [1, 0, 1],
                          'pred_label' : ['Y', 'X', 'Y'],
                          ll_name : ll_exp}, 
                          index = [0, 1, 2])
    obs = mll_testset(XY, test, unc, lbls)
    assert obs.equals(exp)

def test_mll_testset_ext(dfXY):
    XY, unc, lbls, ll_name = dfXY
    test = pd.DataFrame({'feature' : [4], 
                       'label' : ['W']},
                       index = ['A'])
    ll_exp = [calc_ll_exp(1, 3)]
    exp = pd.DataFrame({'sim_idx' : ['A'],
                          'label' : ['W'],
                          'pred_idx' : [2],
                          'pred_label' : ['Z'],
                          ll_name : ll_exp}, 
                          index = [0])
    obs = mll_testset(XY, test, unc, lbls)
    assert obs.equals(exp)

def test_calc_errors():
    pred_df = pd.DataFrame({'sim_idx' : ['A', 'B'],
                            'NumLabel' : [1.2, 7.5],
                            'Reactor' : ['X', 'Y'],
                            'pred_idx' : [0, 1],
                            'pred_NumLabel' : [0.2, 10],
                            'pred_Reactor' : ['X', 'Z'],
                            'LogLikelihood_xx' : [1, 2]}, 
                            index = [0, 1])
    true_lbls = ['Reactor', 'NumLabel']
    exp = pred_df.copy()
    exp['Reactor_Score'], exp['NumLabel_Error'] = [[True, False], [1, 2.5]]
    obs = calc_errors(pred_df, true_lbls)
    assert obs.equals(exp)

def test_parse_args():
    argv1 = []
    args = parse_args(argv1)
    obs = [args.sim_unc, args.ratios]
    exp = [0.05, False]
    assert obs == exp
    argv2 = ['-unc', '0.1', '-r', '-test', 'yy', '-train', 'xx', '-o', 'zz']
    args = parse_args(argv2)
    obs = [args.sim_unc, args.train_db, args.test_db, args.ratios, args.outfile]
    exp = [0.1, 'xx', 'yy', True, 'zz']
    assert obs == exp
    
