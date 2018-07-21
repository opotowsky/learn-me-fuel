
            d1 = []
            f1 = []
            d2 = []
            f2 = []
            it = []
            a1 = []
            a2 = []
            l1 = []
            l2 = []
            for i in range(0, 10):
                # CV search the hyperparams
                # alg1
                alg1_grid = {"max_depth": np.linspace(3, 90).astype(int), 
                             "max_features": np.linspace(5, len(trainXY.columns)-6).astype(int)}
                alg1_opt = RandomizedSearchCV(estimator=alg1_init, param_distributions=alg1_grid, 
                                              n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                              return_train_score=True)
                alg1_opt.fit(trainX, trainY)
                alg1_init = alg1_opt.best_estimator_
                d1.append(alg1_opt.best_params_['max_depth'])
                f1.append(alg1_opt.best_params_['max_features'])
                
                # alg2
                alg2_grid = alg1_grid
                alg2_opt = RandomizedSearchCV(estimator=alg2_init, param_distributions=alg2_grid,
                                              n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                              return_train_score=True)
                alg2_opt.fit(trainX, trainY)
                alg2_init = alg2_opt.best_estimator_
                d2.append(alg2_opt.best_params_['max_depth'])
                f2.append(alg2_opt.best_params_['max_features'])
                
                # alg3
                alg3_grid = {'n_iter': np.linspace(50, 1000).astype(int), 
                             'alpha_1': np.logspace(-8, 2), 'alpha_2' : np.logspace(-8, 2), 
                             'lambda_1': np.logspace(-8, 2), 'lambda_2' : np.logspace(-8, 2)}
                if Y is not 'r':
                    alg3_opt = RandomizedSearchCV(estimator=alg3_init, param_distributions=alg3_grid,
                                                  n_iter=20, scoring=score, n_jobs=-1, cv=kfold, 
                                                  return_train_score=True)
                    alg3_opt.fit(trainX, trainY)
                    alg3_init = alg3_opt.best_estimator_
                    it.append(alg3_opt.best_params_['n_iter'])
                    a1.append(alg3_opt.best_params_['alpha_1'])
                    a2.append(alg3_opt.best_params_['alpha_2'])
                    l1.append(alg3_opt.best_params_['lambda_1'])
                    l2.append(alg3_opt.best_params_['lambda_2'])

            # Save dat info
            param_file = 'trainset_' + trainset + '_hyperparameters_alt-algs.txt'
            with open(param_file, 'a') as pf:
                pf.write('The following parameters are best from the randomized search for the {} parameter prediction:\n'.format(parameter))
                pf.write('max depth for dtree is {}\n'.format(np.mean(d1)))
                pf.writelines(["%s, " % item  for item in d1])
                pf.write('\nmax features for dtree is {}\n'.format(np.mean(f1))) 
                pf.writelines(["%s, " % item  for item in f1])
                pf.write('\nmax depth for xtree is {}\n'.format(np.mean(d2)))
                pf.writelines(["%s, " % item  for item in d2])
                pf.write('\nmax features for xtree is {}\n'.format(np.mean(f2))) 
                pf.writelines(["%s, " % item  for item in f2])
                if Y is not 'r':
                    pf.write('\nnum iterations for bayes reg is {}\n'.format(np.mean(it)))
                    pf.writelines(["%s, " % item  for item in it])
                    pf.write('\nalpha 1 for bayes reg is {}\n'.format(np.mean(a1)))
                    pf.writelines(["%s, " % item  for item in a1])
                    pf.write('\nalpha 2 for bayes reg is {}\n'.format(np.mean(a2)))
                    pf.writelines(["%s, " % item  for item in a2])
                    pf.write('\nlambda 1 for bayes reg is {}\n'.format(np.mean(l1)))
                    pf.writelines(["%s, " % item  for item in l1])
                    pf.write('\nlambda 2 for bayes reg is {}\n'.format(np.mean(l2)))
                    pf.writelines(["%s, " % item  for item in l2])
