#! /usr/bin/env python3

import os
import subprocess
from all_jobs import parent_jobs, kid_jobs

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
    Reads all the job descriptions from all_jobs.py and makes directories for
    each job for HTC output
    
    """
    for parent_job in parent_jobs:
        make_dirs(parent_job['parent_dir'], kid_jobs['job_dirs'])
    return
    
if __name__ == "__main__":
    main()
