from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
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
    
    # Predictions
    predict1 = cross_val_predict(l1, train.nuc_concs, train.reactor, cv = 5)
    predict2 = cross_val_predict(l2, train.nuc_concs, train.reactor, cv = 5)
    predict3 = cross_val_predict(rc, train.nuc_concs, train.reactor, cv = 5)
    y = train.reactor
    print(metrics.classification_report(y, predict1))
    print(metrics.classification_report(y, predict2))
    print(metrics.classification_report(y, predict3))
    
    # TODO: ROC plots
    #fig, ax = plt.subplots()
    #ax.scatter(y, predict3)
    #ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 2)
    #ax.set_xlabel('Measured')
    #ax.set_ylabel('Predicted')
    #plt.show()

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
    
    # Predictions
    predict1 = cross_val_predict(l1, train.nuc_concs, train.enrichment, cv = 5)
    predict2 = cross_val_predict(l2, train.nuc_concs, train.enrichment, cv = 5)
    predict3 = cross_val_predict(rr, train.nuc_concs, train.enrichment, cv = 5)
    y = train.enrichment
    #print(metrics.mean_absolute_error(expected, predict1))
    #print(metrics.mean_absolute_error(expected, predict2))
    #print(metrics.mean_absolute_error(expected, predict3))

    # Plots
    fig, ax = plt.subplots()
    ax.scatter(y, predict3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

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

    # Predictions
    predict1 = cross_val_predict(l1, train.nuc_concs, train.burnup, cv = 5)
    predict2 = cross_val_predict(l2, train.nuc_concs, train.burnup, cv = 5)
    predict3 = cross_val_predict(rr, train.nuc_concs, train.burnup, cv = 5)
    y = train.burnup
    #print(metrics.mean_absolute_error(expected, predict1))
    #print(metrics.mean_absolute_error(expected, predict2))
    #print(metrics.mean_absolute_error(expected, predict3))
    #print(metrics.r2_score(expected, predict1))
    #print(metrics.r2_score(expected, predict2))
    #print(metrics.r2_score(expected, predict3))
    
    # Plots
    fig, ax = plt.subplots()
    ax.scatter(y, predict3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    return

