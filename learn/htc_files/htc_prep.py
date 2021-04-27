#! /usr/bin/env python3

import csv

def make_paramstxt(train, txtfile, file_descrip):
    
    rxtr_param = ['reactor', 'cooling', 'enrichment', 'burnup']
    algs = ['knn', 'dtree']
    # all funcs:
    #func_type = ['--track_preds', '--err_n_scores', '--learn_curves', 
    #             '--valid_curves', '--test_compare', '--random_error']
    # funcs ran for nuc conc scenario:
    #func_type = ['--test_compare', '--random_error']
    # funcs ran for gamma spec scenario:
    func_type = ['--err_n_scores',]
    #tset_frac = [0.1, 0.3, 1.0]
    #tset_frac = [0.2, 0.6]
    tset_frac = [0.1, 0.2, 0.3, 0.5, 1.0]
    cv = 5

    for param in rxtr_param:
        for func in func_type:
            # test_compare or blank args
            # only applicable in nuc conc scenario so leaving as-is
            if 'test_compare' in func:
                filename = 'sfco_compare' + txtfile
                ts_flag = '--testing_set'
                if '15' in train:
                    testset = 'sfcompo_nuc15.pkl'
                else:
                    testset = 'sfcompo_nuc29.pkl'
            else:
                filename = func[2:] + txtfile
                ts_flag = ' '
                testset = ' '
            for alg in algs:
                for frac in tset_frac:
                    # outfile naming
                    outfile = param + '_' + alg + '_tset' + str(frac) + file_descrip
                    with open(filename, 'a') as f:
                        w = csv.writer(f)
                        job = [outfile, param, alg, str(frac), str(cv),
                               train, func, ts_flag, testset]
                        w.writerow(job)
    return

def main():
    """
    Populates the necessary htc param.txt files
    
    """
    # nuclide concentration lists
    #dblist = ['sim_grams_nuc15.pkl', 'sim_grams_nuc29.pkl']
    #txtfile = ['_nuc15_param.txt', '_nuc29_param.txt']
    #outfile = ['_nuc15', '_nuc29']
    
    # activity lists
    dblist = ['d1_hpge_spectra_peaks_trainset.pkl',
              'd2_hpge_spectra_peaks_trainset.pkl',
              ]
    txtfile = ['_d1_param.txt',
               '_d2_param.txt',
               ]
    outfile = ['_d1_hpge',
               '_d2_hpge',
               ]

    for i, traindb in enumerate(dblist):
        make_paramstxt(traindb, txtfile[i], outfile[i])
    return
    
if __name__ == "__main__":
    main()
