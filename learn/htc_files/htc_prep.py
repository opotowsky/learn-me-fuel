#! /usr/bin/env python3

import csv

def make_paramstxt(train, txtfile):
    
    rxtr_param = ['reactor', 'cooling', 'enrichment', 'burnup']
    algs = ['knn', 'dtree', 'svm']
    #func_type = ['--track_preds', '--err_n_scores', '--learn_curves', 
    #             '--valid_curves', '--test_compare', '--random_error']
    func_type = ['--random_error', '--test_compare']
    tset_frac = 0.1
    cv = 5

    for param in rxtr_param:
        for func in func_type:
            # test_compare or blank args
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
                # outfile naming
                if '15' in train:
                    outfile = param + '_' + alg + '_' + str(tset) + 'tset_nuc15'
                else:
                    outfile = param + '_' + alg + '_' + str(tset) + 'tset_nuc29'
                with open(filename, 'a') as f:
                    w = csv.writer(f)
                    job = [outfile, param, alg, str(tset_frac), str(cv),
                           train, func, ts_flag, testset]
                    w.writerow(job)
    return

def main():
    """
    Populates the necessary param_nucXX.txt files
    
    """
    train_db = ['sim_grams_nuc15.pkl', 'sim_grams_nuc29.pkl']
    txtfile = ['_nuc15_param.txt', '_nuc29_param.txt']
    
    for train, tfile in zip(train_db, txtfile):
        make_paramstxt(train, tfile)
    return
    
if __name__ == "__main__":
    main()
