# zdroj: http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

# prakticky vypocet tunovania parametrov pre GBM
# pozor, vypocet dlho trva, spustaj s rozumom

# priklad pouzita GBM v sutazi(vitazne riesenie): https://github.com/analyticsvidhya/DateYourData/blob/master/Shan_Modeling.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import time

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4



t1 = time.time()



train = pd.read_csv('Dataset/train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'



def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()




### baseline model with default parameters
##
###Choose all predictors except target & IDcols
##predictors = [x for x in train.columns if x not in [target, IDcol]]
##gbm0 = GradientBoostingClassifier(random_state=10)
##modelfit(gbm0, train, predictors)





#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]


# ideme zistovant number of trees. Ostatne sme zvolili predbezne a intuitivne.
# n_jobs mi na tomto PC funguje len ked je 1
##param_test1 = {'n_estimators':list(range(50,61,10))}
##estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
##gsearch1 = GridSearchCV(estimator = estimator, param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
##gsearch1.fit(train[predictors],train[target])
##print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
##
# vyslo nam n_estimators=60 ako najlepsie




# dalej pokracujem s tunovanim max_depth a min_samples_split, kedze su vyznamnejsie ako ostatne.

##param_test2 = {'max_depth':list(range(7,10,2)), 'min_samples_split':list(range(800,1001,200))}
##gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
##param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
##gsearch2.fit(train[predictors],train[target])
##print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

# vyslo nam max_depth=9 a min_samples_split=1000




##param_test3 = {'min_samples_split':list(range(1000,1201,200)), 'min_samples_leaf':list(range(50,71,10))}
##gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10), 
##param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
##gsearch3.fit(train[predictors],train[target])
##print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

# min_samples_leaf=60 a min_samples_split=1200




# vyuzijeme najlepsi estimator z posledneho tuningu na vykreslenie feature importance. Vidime ze sa
# vyuziva oproti baseline modelu viac features a su rozumnejsie rozdelene.
#####modelfit(gsearch3.best_estimator_, train, predictors)

##estm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1200,min_samples_leaf=60,max_depth=9,max_features='sqrt',subsample=0.8,random_state=10)
##modelfit(estm, train, predictors)




# tunujeme posledne parametre, zacneme napr. s max_features
##param_test4 = {'max_features':list(range(7,10,2))}
##gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),
##param_grid = param_test4, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
##gsearch4.fit(train[predictors],train[target])
##print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

# vybrali sme max_features=7



##param_test5 = {'subsample':[0.8,0.85]}
##gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
##param_grid = param_test5, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
##gsearch5.fit(train[predictors],train[target])
##print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)

# vybrali sme vraj subsample=0.85








# oni to porovnaval este aj na privatnom datasete

# teraz skusime znizit learning rate a zvysit pocet stromov

##gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
##modelfit(gbm_tuned_1, train, predictors)




##gbm_tuned_3 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
##warm_start=True)
##modelfit(gbm_tuned_3, train, predictors, performCV=False)



t2 = time.time()

print('trvanie vypoctu:',t2-t1,'sek.')
