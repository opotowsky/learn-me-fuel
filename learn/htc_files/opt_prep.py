#! /usr/bin/env python3

import csv

def make_paramstxt(train, db_desc, txtfile):
    
    rxtr_param = ['reactor', 'cooling', 'enrichment', 'burnup']
    algs = ['knn', 'dtree']#, 'svm']
    tset_frac = 1.0
    cv = 5

    with open(txtfile, 'a') as f:
        for param in rxtr_param:
            for alg in algs:
                w = csv.writer(f)
                job = [param, alg, db_desc, str(tset_frac), str(cv), train]
                w.writerow(job)
    return

def main():
    """
    Populates the necessary params_optimize_nucXX.txt files
    
    """
    # nuclide concentration lists
    #train_db = ['sim_grams_nuc29.pkl']
    #db_descs = ['nuc29']
    #txtfile = ['optimize_nuc29_param.txt']

    # processed spectra lists
    train_db = ['nuc7_activities_scaled_1g_reindex.pkl',
                'nuc12_activities_scaled_1g_reindex.pkl',
                'nuc32_activities_scaled_1g_reindex.pkl',
                'd1_hpge_spectra_short_peaks_trainset.pkl',
                'd1_hpge_spectra_long_peaks_trainset.pkl',
                'd1_hpge_spectra_auto_peaks_trainset.pkl',
                'd2_hpge_spectra_short_peaks_trainset.pkl',
                'd2_hpge_spectra_long_peaks_trainset.pkl',
                'd2_hpge_spectra_auto_peaks_trainset.pkl',
                'd3_czt_spectra_short_peaks_trainset.pkl',
                'd3_czt_spectra_long_peaks_trainset.pkl',
                'd3_czt_spectra_auto_peaks_trainset.pkl',
                'd4_nai_spectra_short_peaks_trainset.pkl',
                'd4_nai_spectra_long_peaks_trainset.pkl',
                'd4_nai_spectra_auto_peaks_trainset.pkl',
                'd5_labr3_spectra_short_peaks_trainset.pkl',
                'd5_labr3_spectra_long_peaks_trainset.pkl',
                'd5_labr3_spectra_auto_peaks_trainset.pkl',
                'd6_sri2_spectra_short_peaks_trainset.pkl',
                'd6_sri2_spectra_long_peaks_trainset.pkl',
                'd6_sri2_spectra_auto_peaks_trainset.pkl',
                ]
    db_descs = ['act7', 'act12', 'act32', 
                'd1_short', 'd1_long', 'd1_auto',
                'd2_short', 'd2_long', 'd2_auto',
                'd3_short', 'd3_long', 'd3_auto',
                'd4_short', 'd4_long', 'd4_auto',
                'd5_short', 'd5_long', 'd5_auto',
                'd6_short', 'd6_long', 'd6_auto',
                ]
    txtfile = 'optimize_dets_param.txt'

    for train, db_desc in zip(train_db, db_descs):
        make_paramstxt(train, db_desc, txtfile)
    return
    
if __name__ == "__main__":
    main()
