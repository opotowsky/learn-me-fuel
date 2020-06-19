#! /usr/bin/env python3

from mll_calc.mll_calc import format_XY, ratios, get_pred, mll_testset, parse_args
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
    ext_test = False
    test = XY.copy()
    ll_exp = [calc_ll_exp(1, 2), calc_ll_exp(1, 1), calc_ll_exp(1, 2)]
    exp = pd.DataFrame({'sim_idx' : [0, 1, 2],
                          'label' : ['X', 'Y', 'Z'],
                          'pred_idx' : [1, 0, 1],
                          'pred_label' : ['Y', 'X', 'Y'],
                          ll_name : ll_exp}, 
                          index = [0, 1, 2])
    obs = mll_testset(XY, test, ext_test, unc, lbls)
    assert obs.equals(exp)

def test_mll_testset_ext(dfXY):
    XY, unc, lbls, ll_name = dfXY
    ext_test = True
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
    obs = mll_testset(XY, test, ext_test, unc, lbls)
    assert obs.equals(exp)

@pytest.mark.parametrize('argv, exp',
                         [(['0.05', 'xx', 'yy', 'zz', 'dir', '0', '1', '--ext-test', '--ratios'], 
                           [0.05, 'xx', 'yy', 'zz', 'dir', [0, 1], True, True]
                           ),
                          (['0.05', 'xx', 'yy', 'zz', 'dir', '0', '1', '--no-ext-test', '--no-ratios'], 
                           [0.05, 'xx', 'yy', 'zz', 'dir', [0, 1], False, False]
                           )
                          ]
                         )
def test_parse_args(argv, exp):
    args = parse_args(argv)
    obs = [args.sim_unc, args.train_db, args.test_db, args.outfile, args.outdir, args.db_rows, args.ext_test, args.ratios]
    assert obs == exp
