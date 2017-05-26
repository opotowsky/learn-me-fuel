from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn import metrics


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
    print(metrics.classification_report(expected, predict1))
    print(metrics.classification_report(expected, predict2))
    print(metrics.classification_report(expected, predict3))
    
    return

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
    print(metrics.mean_absolute_error(expected, predict1))
    print(metrics.mean_absolute_error(expected, predict2))
    print(metrics.mean_absolute_error(expected, predict3))

    return

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
    print(metrics.mean_absolute_error(expected, predict1))
    print(metrics.mean_absolute_error(expected, predict2))
    print(metrics.mean_absolute_error(expected, predict3))
    print(metrics.r2_score(expected, predict1))
    print(metrics.r2_score(expected, predict2))
    print(metrics.r2_score(expected, predict3))
    
    return

