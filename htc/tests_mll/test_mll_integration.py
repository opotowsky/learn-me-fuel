#! /usr/bin/env python3

import csv
import pytest
import subprocess
import numpy as np
import pandas as pd

@pytest.mark.parametrize('exp, argv',
                         [(11, ['-train', './tests_mll/sample10sim.pkl']), 
                          (3, ['-train', './tests_mll/sample10sim.pkl', 
                               '-e', '-test', './tests_mll/sfcompo2.pkl'
                               ]
                           )
                          ]
                         )
def test_integration(tmpdir, exp, argv):
    outfile = tmpdir.join('output.csv')
    cmd_list = ['./mll_calc/mll_calc.py'] + argv + ['-o', outfile]
    subprocess.run(cmd_list)
    # Just testing # of lines in final output for now
    with open(outfile, 'r') as f: 
        reader = csv.reader(f)
        obs_lines = len(list(reader))
    assert obs_lines == exp
