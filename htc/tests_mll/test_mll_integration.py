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

#def test_unaltered_db(tmpdir):
#    
#    XY = pd.read_pickle(args.train_db)
#    XY.reset_index(inplace=True, drop=True)
#    if 'total' in XY.columns:
#        XY.drop('total', axis=1, inplace=True)
#    XY = XY.loc[XY['Burnup'] > 0]
#
#    #### TO REMOVE ####
#    # small db for testing code
#    #XY = XY.sample(50)
#    XY = XY.head(25)
#    #### END REMOVE ####
#    # in-script test to ensure the db wasn't altered:
#    if XY != pd.read_pickle(args.train_db):
#        sys.exit('Training DB no longer matches')
#
