#! /usr/bin/env python3

import csv
import pytest
import subprocess
import numpy as np
import pandas as pd

@pytest.mark.parametrize('exp, bool_argv',
                         [(11, ['--no-ext-test', '--no-ratios']),
                          (3, ['--ext-test', '--no-ratios']),
                          (11, ['--no-ext-test', '--ratios']),
                          (3, ['--ext-test', '--ratios'])
                          ]
                         )
def test_integration(tmpdir, exp, bool_argv):
    const_argv = ['0.05', './tests_mll/sample10sim.pkl', './tests_mll/sfcompo2.pkl'] 
    htc_argv = ['JobDir', '0', '10']
    outfile = tmpdir.join('output')
    cmd_list = ['./mll_calc/mll_calc.py'] + const_argv + [outfile] + htc_argv + bool_argv
    subprocess.run(cmd_list)
    # Just testing # of lines in final output for now
    with open(outfile + '.csv', 'r') as f: 
        reader = csv.reader(f)
        obs_lines = len(list(reader))
    assert obs_lines == exp

