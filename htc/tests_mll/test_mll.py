#! /usr/bin/env python3

from mll_calc.mll_calc import *

import math
#import os
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
    XY = pd.DataFrame({'A' : [1., 2., 3.], 
                       'B' : [1., 1., 0.],
                       'label' : [1, 1, 1]})
    exp = pd.DataFrame({'A/B' : [1., 2., 0.], 
                        'B/A' : [1., 0.5, 0.],
                        'label' : [1, 1, 1]})
    obs = ratios(XY, ratio_list, labels)
    assert obs.equals(exp)
