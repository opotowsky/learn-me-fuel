#! /usr/bin/env python3

import sys
import csv
import numpy as np
import pandas as pd
from all_jobs import parent_jobs, kid_jobs, train_test_pkls

def format_db(db_path):
    # TODO copied from mll_calc.py to save time
    XY = pd.read_pickle(db_path)
    if 'total' in XY.columns:
        XY.drop('total', axis=1, inplace=True)
    XY = XY.loc[XY['Burnup'] > 0]
    XY.reset_index(inplace=True, drop=True)
    return XY

def row_calcs(ext_test, n_rows):
    ################################################
    # local filepaths corresponding to DBs in HTC: #
    tamu_db = '~/sims_n_results/nucmoles_opusupdate_aug2019/not-scaled_15nuc.pkl'
    sfco_db = '~/sfcompo/format_clean/sfcompo_formatted.pkl'
    ################################################
    
    if 'no' in ext_test:
        test_set = format_db(tamu_db)
    else:
        test_set = format_db(sfco_db)
    db_rows = len(test_set.index)
    init_rows = np.arange(0, db_rows, n_rows).tolist()
    end_rows = init_rows[1:]
    end_rows.append(db_rows+1)
    
    ################################################
    ################ In-script test ################
    ################################################
    total_jobs = db_rows // n_rows + 1
    if len(init_rows) != total_jobs or len(end_rows) != total_jobs:
        sys.exit('total expected jobs does not equal one of db_row lists')
    ################################################

    return init_rows, end_rows

def make_paramstxt(parent_job, kid_jobs):
    init_rows, end_rows = row_calcs(parent_job['ext_test'], kid_jobs['rows_per_job'])
    parent_dir = parent_job['parent_dir']
    fname = parent_dir + '_params.txt'
    with open(fname, 'w') as f:
        w = csv.writer(f) 
        for kid_dir, unc in zip(kid_jobs['job_dirs'], kid_jobs['uncs']):
            job_dir = parent_dir + '/' + kid_dir
            for i in range(0, len(init_rows)):
                job = [job_dir, unc, 
                       train_test_pkls[0], train_test_pkls[1], 
                       str(i).zfill(3), init_rows[i], end_rows[i],
                       parent_job['ext_test'], parent_job['ratios']
                       ]
                w.writerow(job)    
    return

def main():
    """
    Reads all the job descriptions from all_jobs.py and populates the necessary
    params_mll_calc.txt files
    
    """
    for parent_job in parent_jobs:
        make_paramstxt(parent_job, kid_jobs)
    return
    
if __name__ == "__main__":
    main()
