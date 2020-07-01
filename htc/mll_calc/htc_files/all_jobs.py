#! /usr/bin/env python3

uncs = [0.05] #[0.05, 0.1, 0.15, 0.2]
job_dirs = ['Job' str(unc) for unc in uncs]
pkls = ['sim_tamu_nucmols.pkl', 'sfcompo.pkl']

tamu_jobs = {'job_dirs' : [],
             'uncs' : uncs,
             'pkls' : pkls,
             'ext_test' : ['--no-ext-test'],
             'ratios' : ['--no-ratios', '--ratios']
             }

scfo_jobs = {'job_dirs' : [],
             'uncs' : uncs,
             'pkls' : pkls,
             'ext_test' : ['--ext-test'],
             'ratios' : ['--ratios']
             }

