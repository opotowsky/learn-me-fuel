#! /usr/bin/env python3

import csv
import numpy as np

def make_paramstxt(train, txtfile, file_descrip):
    
    rxtr_param = ['reactor', 'cooling', 'enrichment', 'burnup']
    algs = ['knn', 'dtree']
    tset_frac = [1.0]
    cv = [5]
    
    # all funcs:
    #func_type = ['--int_test_compare', '--err_n_scores', '--learn_curves', 
    #             '--valid_curves', '--ext_test_compare', '--random_error',
    #             '--cv_pred']
    # funcs ran for nuc conc scenario: ['--ext_test_compare', '--random_error']
    # funcs ran for gamma spec scenario: ['--err_n_scores', '--int_test_compare', --cv_pred]
    
    # changes unique to special sfco expmt
    #func_type = ['--ext_test_compare']
    #tset_frac = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    #outfile = param + '_' + alg + '_idx' + str(int(frac)) + '_df' + str(cv)  + file_descrip

    # for "learning curve" expmt
    #fracs = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 
    #                  0.2, 0.23, 0.25, 0.27, 0.3])
    #tset_frac = np.around(1 + fracs, 2)
    
    # next mini exmpt, high CV to have 15 7% test sets:
    frac = 1.0
    cvs = [5, 10, 15]
    func_type = ['--cv_pred']

    for param in rxtr_param:
        for func in func_type:
            # only applicable in nuc conc scenario so leaving as-is
            testset = 'null'
            if 'ext_test_compare' in func:
                filename = 'sfco_compare' + txtfile
                testset = 'sfcompo_nuc29.pkl'
            elif 'int_test_compare' in func:
                filename = 'mimic_mll' + txtfile
            else:
                filename = func[2:] + txtfile
            for alg in algs:
                #for frac in tset_frac:
                for cv in cvs:
                    #outfile = param + '_' + alg + '_tset' + str(frac) + file_descrip
                    outfile = param + '_' + alg + '_tset' + str(frac) + '_cv' + str(cv)  + file_descrip
                    with open(filename, 'a') as f:
                        w = csv.writer(f)
                        job = [outfile, param, alg, str(frac), str(cv),
                               train, func, '--testing_set', testset]
                        w.writerow(job)
    return

def main():
    """
    Populates the necessary htc param.txt files
    
    """
    # nuclide concentration lists
    dblist = ['sim_grams_nuc29.pkl']
    txtfiles = ['_nuc29_param.txt']
    outfiles = ['_nuc29']
    #for i, traindb in enumerate(dblist):
    #    make_paramstxt(traindb, txtfiles[i], outfiles[i])
    
    # activity lists
    dblist = ['d1_hpge_spectra_31peaks_trainset.pkl',
              'd1_hpge_spectra_113peaks_trainset.pkl',
              'd1_hpge_spectra_auto_peaks_trainset.pkl',
              'd2_hpge_spectra_31peaks_trainset.pkl',
              'd2_hpge_spectra_113peaks_trainset.pkl',
              'd2_hpge_spectra_auto_peaks_trainset.pkl',
              'd3_czt_spectra_31peaks_trainset.pkl',
              'd3_czt_spectra_113peaks_trainset.pkl',
              'd3_czt_spectra_auto_peaks_trainset.pkl',
              'd4_nai_spectra_31peaks_trainset.pkl',
              'd4_nai_spectra_113peaks_trainset.pkl',
              'd4_nai_spectra_auto_peaks_trainset.pkl',
              'd5_labr3_spectra_31peaks_trainset.pkl',
              'd5_labr3_spectra_113peaks_trainset.pkl',
              'd5_labr3_spectra_auto_peaks_trainset.pkl',
              'd6_sri2_spectra_31peaks_trainset.pkl',
              'd6_sri2_spectra_113peaks_trainset.pkl',
              'd6_sri2_spectra_auto_peaks_trainset.pkl',
              ]
    txtfile = ['_n31_param.txt',
               '_n113_param.txt',
               '_auto_param.txt',
               ]
    txtfiles = txtfile * 6
    outfile = ['_d1_hpge',
               '_d2_hpge',
               '_d3_czt',
               '_d4_nai',
               '_d5_labr3',
               '_d6_sri2',
               ]
    outfiles = [i for i in outfile for j in range (0, 3)]
    
    for i, traindb in enumerate(dblist):
        make_paramstxt(traindb, txtfiles[i], outfiles[i])
    
    dblist2 = ['nuc4_activities_scaled_1g_reindex.pkl',
               'nuc9_activities_scaled_1g_reindex.pkl',
               'nuc32_activities_scaled_1g_reindex.pkl',
               'sim_grams_nuc29.pkl',
               ]
    txtfiles = ['_act4_param.txt',
                '_act9_param.txt',
                '_act32_param.txt',
                '_nuc29_param.txt',
                ]
    outfiles = ['_act4',
                '_act9',    
                '_act32',    
                '_nuc29',    
                ]
    
    for i, traindb in enumerate(dblist2):
        make_paramstxt(traindb, txtfiles[i], outfiles[i])
    

    return
    
if __name__ == "__main__":
    main()
