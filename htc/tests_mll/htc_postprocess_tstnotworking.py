#! /usr/bin/env python3

from mll_calc.htc_postprocess import calc_errors
import pandas as pd

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

