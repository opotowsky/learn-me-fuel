from learn.tools import splitXY, track_predictions, errors_and_scores, validation_curves, learning_curves, ext_test_compare
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import pytest
import os

# For now, skipping pandas df-related functions; unsure how to test this
n_obs = 500
n_feats = 50

def data_setup():
    X, y = make_regression(n_samples=n_obs, n_features=n_feats, noise=.2)
    # mimics output of splitXY
    X = pd.DataFrame(X, index=np.arange(0, n_obs), columns=np.arange(0, n_feats))
    y = pd.Series(y)
    score = 'explained_variance'
    kfold = KFold(n_splits=5, shuffle=True)
    alg1_init = KNeighborsRegressor(weights='distance')
    alg2_init = DecisionTreeRegressor()
    alg3_init = SVR()
    csv_name = 'test.csv'
    return X, y, score, kfold, alg1_init, alg2_init, alg3_init, csv_name

def test_track_predictions(tmpdir):
    X, y, _, kfold, alg1_init, alg2_init, alg3_init, csv_name = data_setup()
    csv_name = tmpdir.join(csv_name)
    track_predictions(X, y, alg1_init, alg2_init, alg3_init, kfold, csv_name, X)

def test_errors_and_scores(tmpdir):
    X, y, score, kfold, alg1_init, alg2_init, alg3_init, csv_name = data_setup()
    csv_name = tmpdir.join(csv_name)
    scores = [score,]
    errors_and_scores(X, y, alg1_init, alg2_init, alg3_init, scores, kfold, csv_name)

def test_validation_curves(tmpdir):
    X, y, score, kfold, alg1_init, alg2_init, alg3_init, csv_name = data_setup()
    csv_name = tmpdir.join(csv_name)
    learning_curves(X, y, alg1_init, alg2_init, alg3_init, kfold, score, csv_name)

def test_learning_curves(tmpdir):
    X, y, score, kfold, alg1_init, alg2_init, alg3_init, csv_name = data_setup()
    csv_name = tmpdir.join(csv_name)
    validation_curves(X, y, alg1_init, alg2_init, alg3_init, kfold, score, csv_name)

def test_ext_test_compare(tmpdir):
    _, _, _, _, alg1_init, alg2_init, alg3_init, csv_name = data_setup()
    csv_name = tmpdir.join(csv_name)
    # Get real origen data for this test
    trainpath = 'learn/pkl_trainsets/2jul2018/2jul2018_trainset1_'
    pkl = trainpath + 'nucs_fissact_not-scaled.pkl'
    trainXY = pd.read_pickle(pkl)
    trainXY.reset_index(inplace=True, drop=True) 
    trainXY = trainXY.sample(frac=0.1)
    X, rY, cY, eY, bY = splitXY(trainXY)
    ext_test_compare(X, bY, alg1_init, alg2_init, alg3_init, csv_name)
