# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 18:53:50 2016

@author: Pablo

zdroj: http://nbviewer.jupyter.org/github/cs109/2015lab5/blob/master/Classification.ipynb

kNN pre iris dataset 

A) nacitanie dat a klasifikacna funkcia pre zvoleny parameter (iba jeden)
B) tunovanie poctu susedov s vykreslenim priemernych chyb pre kazdu hodnotu
C) vseobecna funkcia na tunovanie




kNN je citlive na nadbytocne features, ktore lahko pridavaju sum a kazia vysledok.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


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

# vykreslenie klasifikacie s pravdepodobnostnymi hranicami. iba pre 2D!!!
def points_plot_prob(ax, Xtr, Xte, ytr, yte, clf, colorscale=cmap_light, cdiscrete=cmap_bold, ccolor=cm, psize=10, alpha=0.1):
    ax,xx,yy = points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=False, colorscale=colorscale, cdiscrete=cdiscrete, psize=psize, alpha=alpha, predicted=True) 
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=ccolor, alpha=.2, axes=ax)
    cs2 = plt.contour(xx, yy, Z, cmap=ccolor, alpha=.6, axes=ax)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14, axes=ax)
    return ax    
    
    
#----------------------------------------A)-----------------------------
   

# import some data to play with
iris = datasets.load_iris()
#X = iris.data[:, 2:] # two features
X = iris.data # all features
y = iris.target



# split sety
Xlr, Xtestlr, ylr, ytestlr = train_test_split(X, y, test_size=0.33)



plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
plt.title('log. regr')
plt.show()


# normovane
Xstd=(X - X.mean(axis=0))/X.std(axis=0)



'''
- rozdelenie na train a test
- zobrazenie skore na train a test sete
- zobrazenie klasifikacie (len ked su vstupne data 2D)
'''
def classify(X,y, nbrs, plotit=True, train_size=0.6):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
    clf= KNeighborsClassifier(nbrs)
    clf=clf.fit(Xtrain, ytrain)
    #in sklearn accuracy can be found by using "score". It predicts and then gets the accuracy
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    Xall=np.concatenate((Xtrain, Xtest))
    print ("Accuracy on training data: %0.2f" % (training_accuracy))
    print ("Accuracy on test data:     %0.2f" % (test_accuracy))
    if plotit:
        plt.figure()
        ax=plt.gca()
        points_plot(ax, Xtrain, Xtest, ytrain, ytest, clf, alpha=0.3, psize=20)
    return nbrs, training_accuracy, test_accuracy


#classify(Xstd,y,9)


#----------------------------------------B)-----------------------------

# ------------Error against complexity (k), and cross-validation------------
#import seaborn as sns
#c0=sns.color_palette()[0]
#c1=sns.color_palette()[1]
#c2=sns.color_palette()[2]

## skusanie rozny pocet susedov a vykreslenie chyby 
## (jeden parameter 200 krat trenovany a priemerovany)
#fits={}
#nbrs=np.arange(1,40,1) # pocet susedov
#for k in nbrs:
#    fits[k]=[]
#    for i in range(200):
#        fits[k].append(classify(Xstd, y,k, False))
#
#fmeanstr = np.array([1.-np.mean([t[1] for t in fits[e]]) for e in nbrs])
#fmeanste = np.array([1.-np.mean([t[2] for t in fits[e]]) for e in nbrs])
#fstdsstr = np.array([np.std([t[1] for t in fits[e]]) for e in nbrs])
#fstdsste = np.array([np.std([t[2] for t in fits[e]]) for e in nbrs])
#
#plt.figure()
#plt.gca().invert_xaxis()
#plt.plot(nbrs, fmeanstr, color=c0, label="training");
#plt.fill_between(nbrs, fmeanstr - fstdsstr, fmeanstr+fstdsstr, color=c0, alpha=0.3)
#plt.plot(nbrs, fmeanste, color=c1, label="testing");
#plt.fill_between(nbrs, fmeanste - fstdsste, fmeanste+fstdsste, color=c1, alpha=0.5)
#
#plt.legend()
#plt.show()

#----------------------------------------C)-----------------------------

# -----------------automatizacia validacie--------------------------------

'''
- split na train a test set
- gridsearch
- vysledok je best estimator a vypise best skore
'''
def cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(Xtrain, ytrain)
    print ("BEST PARAMS", gs.best_params_)
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


# vlozenie dat do dataframeu, aby sa dala vyuzit funkcia do_classify
df = pd.DataFrame(X,columns=['a','b','c','d'])
df['target'] = y


# funkcia na tunovanie
bestcv, Xtrain, ytrain, Xtest, ytest = do_classify(KNeighborsClassifier(), {"n_neighbors": list(range(1,40,2))}, df, ['a','c'], 'target')


print('confusion_matrix:')
print(confusion_matrix(ytest, bestcv.predict(Xtest), ))

# tieto vykreslenie funguju dobre len na 2D datach
if Xtrain.shape[1] == 2: 
    plt.figure()
    ax=plt.gca()
    points_plot(ax, Xtrain, Xtest, ytrain, ytest, bestcv, alpha=0.5, psize=20)
    
    plt.figure()
    ax=plt.gca()
    points_plot_prob(ax, Xtrain, Xtest, ytrain, ytest, bestcv, alpha=0.5, psize=20)

