from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn import metrics
from math import sqrt


def reactor(train, test):
    """
    Training for Reactor Type
    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsClassifier(metric='l1', p=1)
    l2 = KNeighborsClassifier(metric='l2', p=2)
    rc = RidgeClassifier()
    l1.fit(train.nuc_concs, train.reactor)
    l2.fit(train.nuc_concs, train.reactor)
    rc.fit(train.nuc_concs, train.reactor)
    
    # Predictions
    predict1 = l1.predict(test.nuc_concs)
    predict2 = l2.predict(test.nuc_concs)
    predict3 = rc.predict(test.nuc_concs)
    expected = test.reactor
    acc_l1 = metrics.accuracy_score(expected, predict1)
    acc_l2 = metrics.accuracy_score(expected, predict2)
    acc_rc = metrics.accuracy_score(expected, predict3)
    acc_reactor = (acc_l1, acc_l2, acc_rc)

    return acc_reactor

def enrichment(train, test):
    """
    Training for Enrichment
    """
    
    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsRegressor(metric='l1', p=1)
    l2 = KNeighborsRegressor(metric='l2', p=2)
    rr = Ridge()
    l1.fit(train.nuc_concs, train.enrichment)
    l2.fit(train.nuc_concs, train.enrichment)
    rr.fit(train.nuc_concs, train.enrichment)
    
    # Predictions
    predict1 = l1.predict(test.nuc_concs)
    predict2 = l2.predict(test.nuc_concs)
    predict3 = rr.predict(test.nuc_concs)
    expected = test.enrichment
    err_l1 = sqrt(metrics.mean_squared_error(expected, predict1))
    err_l2 = sqrt(metrics.mean_squared_error(expected, predict2))
    err_rr = sqrt(metrics.mean_squared_error(expected, predict3))
    err_enrichment = (err_l1, err_l2, err_rr)

    return err_enrichment

def burnup(train, test):
    """
    Training for Burnup
    """

    # L1 norm is Manhattan Distance
    # L2 norm is Euclidian Distance 
    # Ridge Regression is Linear + L2 regularization
    l1 = KNeighborsRegressor(metric='l1', p=1)
    l2 = KNeighborsRegressor(metric='l2', p=2)
    rr = Ridge()
    l1.fit(train.nuc_concs, train.burnup)
    l2.fit(train.nuc_concs, train.burnup)
    rr.fit(train.nuc_concs, train.burnup)

    # Predictions
    predict1 = l1.predict(test.nuc_concs)
    predict2 = l2.predict(test.nuc_concs)
    predict3 = rr.predict(test.nuc_concs)
    expected = test.burnup
    err_l1 = sqrt(metrics.mean_squared_error(expected, predict1))
    err_l2 = sqrt(metrics.mean_squared_error(expected, predict2))
    err_rr = sqrt(metrics.mean_squared_error(expected, predict3))
    err_burnup = (err_l1, err_l2, err_rr)
    
    return err_burnup

