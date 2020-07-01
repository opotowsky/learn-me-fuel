#! /usr/bin/env python3

import os
import sys
import subprocess
from all_jobs import tamu_jobs, scfo_jobs

def make_paramstxt(jobs):
    """
    """
    
    return

def make_dirs(jobs):
    """
    """
    
    subprocess.run(['mkdir', job_dir])
    return

def main():
    """
    Reads all the job descriptions from all_jobs.py and completes two 
    prep tasks:

    1. Makes directories for each job for HTC output
    2. Populates the necessary params_mll_calc.txt files
    
    """
    for jobs in [tamu_jobs, sfco_jobs]:
        make_dirs(jobs)
        make_paramstxt(jobs)
    return
    
if __name__ == "__main__":
    main()
