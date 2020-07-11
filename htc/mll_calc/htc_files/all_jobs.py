#! /usr/bin/env python3

uncs = [0.05, 0.1, 0.15, 0.2]
job_dirs = ['Job' + str(i) + '_unc' + str(unc) for i, unc in enumerate(uncs)]
train_test_pkls = ['sim_tamu_nucmols.pkl', 'sfcompo.pkl']

kid_jobs = {'job_dirs' : job_dirs,
            'uncs' : uncs,
            'rows_per_job' : 30
            }

tamu_noratio = {'parent_dir' : 'tamuNoRatio',
                'ext_test' : '--no-ext-test',
                'ratios' : '--no-ratios'
                }

tamu_ratio = {'parent_dir' : 'tamuRatio',
              'ext_test' : '--no-ext-test',
              'ratios' : '--ratios'
              }

sfco = {'parent_dir' : 'sfco',
        'ext_test' : '--ext-test',
        'ratios' : '--ratios'
        }

parent_jobs = [tamu_noratio, tamu_ratio, sfco]
