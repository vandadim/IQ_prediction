# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:10:46 2022
@author: vandadim

# This script loads a dataset from a CSV file, where the last column is assumed to be the dependent 
variable (Y) and the other columns are assumed to be independent variables (X). 
# The path to the data file is specified by Data_Path variable. 
# The script reads the data using NumPy's genfromtxt function and separates the data into input
features (Data_X) and output labels (Label).
# The objective of the script is to predict the IQ scores using the input features. 
# The output of the script is the correlation and mean absolute error (MAE) between the predicted IQ scores and actual IQ scores. 
# Usage: 
#     python IQ_pred_Glmnet.py
# 
# Dependencies: 
    - NumPy
    - pandas
    - scikit-learn (for KFold, SVR, RandomizedSearchCV, and mean_absolute_error)
    - scipy.stats (for Pearson correlation coefficient)
#
# Data_Path: path to the CSV file containing the input data and output labels
# Data: NumPy array containing the input data and output labels
# Label: NumPy array containing the output labels (Y)
"""
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVR 
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats


# for data in range (7):
Data_Path = '../Data/Data_ABCD.csv'
Data      = genfromtxt(Data_Path,delimiter=',')
Label     = Data[:,-1]
Data_X    = np.delete(Data, -1, axis=1)
Data_X=stats.zscore(Data_X, axis=0, ddof=1) 
X=Data_X
Y=Label

Save_fold_test=[]
Save_fold_pred=[]


cv = KFold(n_splits=10, shuffle=True, random_state=32)

for train_index, test_index in cv.split(X,y=Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]      
    SVreg = SVR()   
    param_SVR = { 'C':[0.1,1,10,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    SVRgsearch = RandomizedSearchCV(SVreg ,param_distributions = param_SVR, scoring='neg_mean_absolute_error',n_jobs= -1,refit= True, cv=10)

    SVRgsearch.fit(X_train, y_train)
    y_pred_SVR = SVRgsearch.predict(X_test)    
    
    print("Praaaam =",SVRgsearch.best_params_)
    
    print("#######################")
    Corr_bFold = st.pearsonr(y_test, y_pred_SVR)[0]
    print("Correlation_BETWEEN_Fold =",Corr_bFold)
    print("#######################")
    Save_fold_test.append(y_test)
    Save_fold_pred.append(y_pred_SVR)
    
Save_fold_test_Df = pd.DataFrame(Save_fold_test)
Save_fold_pred_Df = pd.DataFrame(Save_fold_pred)
np.savetxt('Y_test_IQ.csv',Save_fold_test_Df)
np.savetxt('Y_Pred_IQ.csv',Save_fold_pred_Df)

All_test =np.concatenate(Save_fold_test)
All_Pred =np.concatenate(Save_fold_pred)

Corr_all = st.pearsonr(All_test, All_Pred)[0]
print("Correlation =",Corr_all)
MAE_all =mean_absolute_error(All_test, All_Pred)
print("Mean absolute error =",MAE_all)