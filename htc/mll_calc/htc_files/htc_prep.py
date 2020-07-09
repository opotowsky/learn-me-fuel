#! /usr/bin/env python3

import os
import csv
import subprocess
from all_jobs import parent_jobs, kid_jobs

def format_db(db_path):
    # TODO copied from mll_calc.py to save time
    XY = pd.read_pickle(db_path)
    if 'total' in XY.columns:
        XY.drop('total', axis=1, inplace=True)
    XY = XY.loc[XY['Burnup'] > 0]
    XY.reset_index(inplace=True, drop=True)
    return XY

def row_calcs(ext_test, pkls, n_rows):
    if 'no' in ext_test:
        test_set = format_db(kid_jobs['pkls'][0])
    else:
        test_set = format_db(kid_jobs['pkls'][1])
    db_rows = len(XY.index)
    
    ########################
    #### In-script test ####
    ########################
    
    total_jobs = db_rows // 10 + 1

    
    ########################

    return init_rows, end_rows

def make_paramstxt(parent_job, kid_jobs):
    init_rows, end_rows = row_calcs(parent_job['ext_test'], kid_jobs['pkls'], kid_jobs['rows_per_job'])
    parent_dir = parent_job['parent_dir']
    fname = parent_dir + '_params.txt'
    with open(fname, 'w') as f:
        w = csv.writer(f) 
        for kid_dir, unc in zip(kid_jobs['job_dirs'], kid_jobs['uncs']):
            job_dir = parent_dir + '/' + kid_dir
            for i in range(0, len(init_row)):
                init_row = init_rows[i]
                end_row = end_rows[i]
                job = [job_dir, unc, 
                       kid_jobs['pkls'][0], kid_jobs['pkls'][1], 
                       init_row, end_row,
                       parent_job['ext_test'], parent_job['ratios']
                       ]
                w.writerow(job)    
    return

def make_dirs(parent_dir, kid_dirs):
    if not os.path.isdir(parent_dir):
        subprocess.run(['mkdir', parent_dir])
    for kid_dir in kid_dirs:
        job_dir = parent_dir + '/' + kid_dir
        if not os.path.isdir(job_dir):
            subprocess.run(['mkdir', job_dir])
    return

def main():
    """
    Reads all the job descriptions from all_jobs.py and completes two 
    prep tasks:

    1. Makes directories for each job for HTC output
    2. Populates the necessary params_mll_calc.txt files
    
    """
    for parent_job in parent_jobs:
        make_dirs(parent_job['parent_dir'], kid_jobs['job_dirs'])
        make_paramstxt(parent_job, kid_jobs)
    return
    
if __name__ == "__main__":
    main()
