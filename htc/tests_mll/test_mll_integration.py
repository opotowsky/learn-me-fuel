#! /usr/bin/env python3

import csv
import pytest
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

@pytest.mark.parametrize('exp, bool_argv',
                         [(10, ['--no-ext-test', '--no-ratios']),
                          (2, ['--ext-test', '--no-ratios']),
                          (10, ['--no-ext-test', '--ratios']),
                          (2, ['--ext-test', '--ratios'])
                          ]
                         )
def test_integration(tmpdir, exp, bool_argv):
    const_argv = ['JobDir', '0.05', './tests_mll/sample10sim.pkl', './tests_mll/sfcompo2.pkl']
    for row in range(0, exp):
        outfile = tmpdir.join('output' + str(row))
        cmd_list = ['./mll_calc/mll_calc.py'] + const_argv + [row] + bool_argv
        subprocess.run(cmd_list)
    # Just testing # of csv files created for now
    assert len(list(Path(tmpdir).glob('*.csv'))) == exp

