# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:10:46 2022

@author: vandadim
"""
import numpy as np
from numpy import genfromtxt
from sklearn import model_selection

import pandas as pd
from sklearn.model_selection import KFold

from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats

for data in range (7):
    Data_Path = '../Data/Data_'+str(data+1)+'.csv'
    Data      = genfromtxt(Data_Path,delimiter=',')
    Label     = Data[:,-1]
    Data_X    = np.delete(Data, -1, axis=1)

    X=Data_X
    Y=Label

    Save_fold_test=[]
    Save_fold_pred=[]


    cv = KFold(n_splits=10, shuffle=True, random_state=32)

    for train_index, test_index in cv.split(X,y=Y):
        X_train, X_test = X[train_index], X[test_index]
        X_train=stats.zscore(X_train, axis=0, ddof=1)
        X_test=stats.zscore(X_test, axis=0, ddof=1)
        y_train, y_test = Y[train_index], Y[test_index]

        xgb = XGBRegressor()
    
        param_XGB = {'nthread':[-1], 'n_estimators':range(3000,6000,1000), 'max_depth':range(6,9,1),'learning_rate':[ 0.01, 0.001], 'min_child_weight':range(1,6,1),'gamma':[i/10.0 for i in range(0,5)],'subsample':[i/10.0 for i in range(1,10)],'colsample_bytree':[i/10.0 for i in range(3,8)]}
        Xgbgsearch = RandomizedSearchCV(xgb ,param_distributions = param_XGB, scoring='neg_mean_absolute_error',refit= True, cv=10)

        # Fitting the models
        Xgbgsearch.fit(X_train,y_train)
        # Predictions of models
    
        y_pred_XGB = Xgbgsearch.predict(X_test)
    
        #Print best param
        print("Praaaam =",Xgbgsearch.best_params_)
    
        # Save Y_test and Y_Pred
        print("#######################")
        Corr_bFold = st.pearsonr(y_test, y_pred_XGB)[0]
        print("Correlation_BETWEEN_Fold =",Corr_bFold)
        print("#######################")
        Save_fold_test.append(y_test)
        Save_fold_pred.append(y_pred_XGB)
    
    Save_fold_test_Df = pd.DataFrame(Save_fold_test)
    Save_fold_pred_Df = pd.DataFrame(Save_fold_pred)
    np.savetxt('Y_test_each_fold_Data_'+str(data+1)+'.csv',Save_fold_test_Df)
    np.savetxt('Y_Pred_each_fold_Data_'+str(data+1)+'.csv',Save_fold_pred_Df)

    All_test =np.concatenate(Save_fold_test)
    All_Pred =np.concatenate(Save_fold_pred)

    Corr_all = st.pearsonr(All_test, All_Pred)[0]
    print("Correlation =",Corr_all)
    MAE_all =mean_absolute_error(All_test, All_Pred)
    print("Mean absolute error =",MAE_all)