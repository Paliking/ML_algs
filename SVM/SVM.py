# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:17:15 2016

@author: Pablo


zdroj: http://nbviewer.jupyter.org/github/cs109/2015lab6/blob/master/lab6-classification-redux.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

##
### import some data to play with
##iris = datasets.load_iris()
##X = iris.data[:, 1:3] # we only take the first two features. We could
## # avoid this ugly slicing by using a two-dim dataset
##y = iris.target
##
### we create an instance of SVM and fit out data. We do not scale our
### data since we want to plot the support vectors
##C = 1.0 # SVM regularization parameter
##svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y)
##
### create a mesh to plot in
##x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##xx, yy = np.meshgrid(np.arange(x_min, x_max),
## np.arange(y_min, y_max))
## 
##plt.subplot(1, 1, 1)
##Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
##Z = Z.reshape(xx.shape)
##plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
##
##
##plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
##plt.xlabel('Sepal length')
##plt.ylabel('Sepal width')
##plt.xlim(xx.min(), xx.max())
##plt.title('SVC with linear kernel')
##plt.show()



# -----------------------B)----------------------------------------

'''
- split na train a test set
- gridsearch
- vysledok je best estimator a vypise best skore
'''
def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print ("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_)
    best = gs.best_estimator_
    return best




'''
Funkcia na tunovanie parametrov. Data sa delia. na trainsete sa robi 5 fold CV.
Najlepsi parameter je potom pouzity na celom training sete.
Vypise sa Accuracy na training a testing sete


VSTUPY
clf - clasifier
parameters - na hladanie v gridsearchu
indf - dataframe s featurmi a targetom spolu
featurenames - list feature nazvov v dataframe, kt. sa maju pouzit
targetname - list target nazvov v dataframe
standardize - ci sa maju normalizovat data
train_size - kolko % ma byt training setu

OUTPUT
clf - najlepsi clasifier fitovany na celom training sete
Xtrain, ytrain, Xtest, ytest - X a y training a testing setu
'''
def do_classify(clf, parameters, indf, featurenames, targetname, standardize=False, train_size=0.7):
    subdf=indf[featurenames]
    if standardize:
        subdfstd=(subdf - subdf.mean())/subdf.std()
    else:
        subdfstd=subdf
    X=subdfstd.values
    y=indf[targetname].values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
    clf = cv_optimize(clf, parameters, Xtrain, ytrain)
    clf=clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print ("Accuracy on training data: %0.2f" % (training_accuracy))
    print ("Accuracy on test data:     %0.2f" % (test_accuracy))
    return clf, Xtrain, ytrain, Xtest, ytest



# vykreslenie klasifikacie iba pre 2D. gulicky su training set a stvorce testing
def points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=True, colorscale=cmap_light, cdiscrete=cmap_bold, alpha=0.1, psize=10, zfunc=False, predicted=False):
    h = .02
    X=np.concatenate((Xtr, Xte))
    if not X.shape[1] == 2:
        print('data su viacrozmerne, nieje mozne ich vykreslit')
        return 'fail'
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    #plt.figure(figsize=(10,6))
    if zfunc:
        p0 = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        p1 = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z=zfunc(p0, p1)
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    ZZ = Z.reshape(xx.shape)
    if mesh:
        plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light, alpha=alpha, axes=ax)
    if predicted:
        showtr = clf.predict(Xtr)
        showte = clf.predict(Xte)
    else:
        showtr = ytr
        showte = yte
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=showtr-1, cmap=cmap_bold, s=psize, alpha=alpha,edgecolor="k")
    # and testing points
    ax.scatter(Xte[:, 0], Xte[:, 1], c=showte-1, cmap=cmap_bold, alpha=alpha, marker="s", s=psize+10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax,xx,yy


# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target


# vlozenie dat do dataframeu, aby sa dala vyuzit funkcia do_classify
df = pd.DataFrame(X,columns=['a','b','c','d'])
df['target'] = y


clfsvm = SVC(kernel="linear")
parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clfsvm, Xtrain, ytrain, Xtest, ytest=do_classify(clfsvm, parameters, df, ['a','c'], 'target')

# tieto vykreslenie funguju dobre len na 2D datach
if Xtrain.shape[1] == 2: 
    plt.figure()
    ax=plt.gca()
    points_plot(ax, Xtrain, Xtest, ytrain, ytest, clfsvm, alpha=0.5)
    plt.show()
