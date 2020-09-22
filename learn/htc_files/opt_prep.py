#! /usr/bin/env python3

import csv

def make_paramstxt(train, txtfile):
    
    rxtr_param = ['reactor', 'cooling', 'enrichment', 'burnup']
    algs = ['knn', 'dtree', 'svm']
    tset_frac = 0.1
    cv = 5

    with open(txtfile, 'w') as f:
        for param in rxtr_param:
            for alg in algs:
                w = csv.writer(f)
                job = [param, alg, str(tset_frac), str(cv), train]
                w.writerow(job)
    return

def main():
    """
    Populates the necessary params_optimize_nucXX.txt files
    
    """
    train_db = ['sim_grams_nuc15.pkl', 'sim_grams_nuc29.pkl']
    txtfile = ['optimize_nuc15_param.txt', 'optimize_nuc29_param.txt']

    for train, tfile in zip(train_db, txtfile):
        make_paramstxt(train, tfile)
    return
    
if __name__ == "__main__":
    main()
