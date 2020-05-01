#! /usr/bin/env python3

from mll_calc.mll_calc import *

import math
#import pytest
#import os
#import pickle
#import numpy as np
import pandas as pd


def test_like_calc():
    y_sim = pd.Series([1, 1, 1, 1])
    y_mes = pd.Series([1, 1, 1, 1])
    std = pd.Series([1, 1, 1, 1])
    exp = (1 / math.sqrt(2 * math.pi)) ** len(y_sim)
    obs = like_calc(y_sim, y_mes, std)
    assert obs == exp

