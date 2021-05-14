#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:04:50 2019

@author: mingyiwang
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

#library for plot
import matplotlib.pyplot as plt
import seaborn as sns

#library for model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Activation

############################
#     data preparation     #
############################
#import
train_salary = pd.read_csv('train_salaries.csv')
train_feature = pd.read_csv('train_features.csv')
test_feature = pd.read_csv('test_features.csv')
list(train_feature)
list(test_feature)
train_feature['jobId'] #from JOB1362684407687 to JOB1362685407686
test_feature['jobId']  #from JOB1362685407687 to JOB1362686407686

#check missing values
train_feature.isnull().any()#no missing value
train_salary.isnull().any() #no missing value
test_feature.isnull().any() #no missing values

##################################
#    Exploratory data analysis   #
##################################
#check skewness of response
plt.hist(train_salary.ix[:,'salary'],bins=20)
plt.savefig("salary.png")
#check extreme values of predictors
train = train_feature.merge(train_salary,on='jobId',how='inner')
plt.subplot(3,3,1)
sns.boxplot(x='companyId',y='salary',data=train)
plt.subplot(3,3,2)
sns.boxplot(x='jobType',y='salary',data=train)
plt.subplot(3,3,3)
sns.boxplot(x='degree',y='salary',data=train)
plt.subplot(3,3,4)
sns.boxplot(x='major',y='salary',data=train)
plt.subplot(3,3,5)
sns.boxplot(x='industry',y='salary',data=train)
plt.subplot(3,3,6)
sns.boxplot(x="yearsExperience",y="salary",data=train)
plt.subplot(3,3,7)
sns.boxplot(x="milesFromMetropolis",y="salary",data=train)
plt.savefig("extreme_value.png")
#select salary=0
train[train['salary']==0]
train_all = train[train.salary != 0]
y=train_all.salary
train_all_x=train_all.ix[:,1:8]
#create dummy variables
predictor_catego = list(train_all)[1:6]
predictor_conti = list(train_all)[6:8]
dummy = pd.get_dummies(train_all[predictor_catego])

train_all_dummy = pd.concat([dummy,train_all[predictor_conti]],axis=1)
y=train_all.salary

#string transfer to integer
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
train_all_x=train_all.ix[:,1:8]
train_all_x = MultiColumnLabelEncoder(columns = list(train_all_x)[0:5]).fit_transform(train_all_x)


############################
#        Modelling         #
############################
#divide data into training set(90%)
# and validation set(10%)
X_dum_train,X_dum_valid,X_train,X_valid,y_train,y_valid  = train_test_split(train_all_dummy,train_all_x,y, test_size=0.1,random_state=123)

"""Normal Linear regression"""
regr = lm.LinearRegression()
regr.fit(X_dum_train,y_train)
lm_train_pred = regr.predict(X_dum_train)
lm_valid_pred = regr.predict(X_dum_valid)
lm_r2_train=r2_score(y_train,lm_train_pred)
lm_mse_train = mean_squared_error(y_train,lm_train_pred)
print('r^2 on train data:%f' %r2_score(y_train,lm_train_pred),'MSE on train data:%f' %mean_squared_error(y_train,lm_train_pred))
#model diagnosis
lm_resid = y_train - lm_train_pred
standard_resid = (lm_resid-np.mean(lm_resid))/np.std(lm_resid)
stats.kstest(standard_resid,'norm') #p-valud=1.36e-240=0

"""Linear regression with Lasso """
cv=10
lasso = lm.LassoCV(n_alphas=50, fit_intercept=True, normalize=True, cv=cv, random_state=123)
lasso.fit(X_dum_train,y_train)
lasso_train_pred = lasso.predict(X_dum_train)
lasso_valid_pred = lasso.predict(X_dum_valid)
lasso_coef = pd.concat([pd.DataFrame(list(train_all_dummy)),pd.DataFrame(lasso.coef_)],axis=1)
lasso_coef.columns = ['coef','value']
lasso_r2_train=r2_score(y_train,lasso_train_pred)
lasso_mse_train=mean_squared_error(y_train,lasso_train_pred)
lasso_r2_valid=r2_score(y_valid,lasso_valid_pred)
lasso_mse_valid=mean_squared_error(y_valid,lasso_valid_pred)
print(lasso_coef[lasso_coef['value']!=0])
print('r^2 on train data:%f' % r2_score(y_train,lasso_train_pred),'MSE on train data:%f' %mean_squared_error(y_train,lasso_train_pred))
print('r^2 on validation data:%f' % r2_score(y_valid,lasso_valid_pred),'MSE on validation data:%f' %mean_squared_error(y_valid,lasso_valid_pred))

##frequency variable plot

"""Ridge regression"""
cv=10
ridge = lm.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],normalize=True,cv=cv)
ridge.fit(X_dum_train,y_train)
ridge_train_pred = ridge.predict(X_dum_train)
ridge_valid_pred = ridge.predict(X_dum_valid)
rg_r2_train=r2_score(y_train,ridge_train_pred)
rg_mse_train=mean_squared_error(y_train,ridge_train_pred)
rg_r2_valid=r2_score(y_valid,ridge_valid_pred)
rg_mse_valid=mean_squared_error(y_valid,ridge_valid_pred)
print('r^2 on train data:%f' % r2_score(y_train,ridge_train_pred),'MSE on train data:%f' %mean_squared_error(y_train,ridge_train_pred))
print('r^2 on validation data:%f' % r2_score(y_valid,ridge_valid_pred),'MSE on validation data:%f' %mean_squared_error(y_valid,ridge_valid_pred))


"""CART"""
cart = DecisionTreeRegressor(max_depth = 40,min_samples_split=20, random_state=123)
dcart=cart.fit(X_train, y_train)
cart_train_pred = cart.predict(X_train)
cart_valid_pred = cart.predict(X_valid)
cart_r2_train=r2_score(y_train,cart_train_pred)
cart_mse_train=mean_squared_error(y_train,cart_train_pred)
cart_r2_valid=r2_score(y_valid,cart_valid_pred)
cart_mse_valid=mean_squared_error(y_valid,cart_valid_pred)
print('r^2 on train data:%f' % cart_r2_train,'MSE on train data:%f' %cart_mse_train)
print('r^2 on validation data:%f' % cart_r2_valid,'MSE on validation data:%f' %cart_mse_valid)

"""Random Forest"""
randomforest = RandomForestRegressor(max_depth=30, min_samples_split=20,  
                           oob_score=True, random_state=123)
randomforest.fit(X_train,y_train)
rf_train_pred = randomforest.predict(X_train)
rf_valid_pred = randomforest.predict(X_valid)
rf_r2_train=r2_score(y_train,rf_train_pred)
rf_mse_train=mean_squared_error(y_train,rf_train_pred)
rf_r2_valid=r2_score(y_valid,rf_valid_pred)
rf_mse_valid=mean_squared_error(y_valid,rf_valid_pred)
print('r^2 on train data:%f' % rf_r2_train,'MSE on train data:%f' % rf_mse_train)
print('r^2 on validation data:%f' % rf_r2_valid,'MSE on validation data:%f' %rf_mse_valid)
#interval prediction

"""GBM"""
#params={
#    'boosting_type': 'gbdt', 
#    'objective': 'regression', 
#    'is_training_metric': True,

#   'learning_rate': 0.1, 
#   'num_leaves': 30, 
#   'max_depth': 20,
#   'min_data_in_leaf': 200,

#   'subsample': 0.8, 
#   'colsample_bytree': 0.8
#     }
#gbm_train = lgb.Dataset(X_train,label=y_train, feature_name=list(X_train.columns), categorical_feature=list(X_train)[0:5])
#gbm_valid = lgb.Dataset(X_valid,label=y_valid, feature_name=list(X_valid.columns), categorical_feature=list(X_valid)[0:5])
#gbm=lgb.train(params, gbm_train,valid_sets=[gbm_train,gbm_valid])
#gbm_train_pred=gbm.predict(X_train)
#gbm_valid_pred=gbm.predict(X_valid)
#gbm_r2_train=r2_score(y_train,gbm_train_pred)
#gbm_mse_train=mean_squared_error(y_train,gbm_train_pred)
#gbm_r2_valid=r2_score(y_valid,gbm_valid_pred)
#gbm_mse_valid=mean_squared_error(y_valid,gbm_valid_pred)
#print('r^2 on train data:%f' % gbm_r2_train,'MSE on train data:%f' % gbm_mse_train)
#print('r^2 on validation data:%f' % gbm_r2_valid,'MSE on validation data:%f' %gbm_mse_valid)


"""Neural Network"""
nn = Sequential()
nn.add(Dense(500,input_dim = X_dum_train.shape[1],activation='relu'))
nn.add(Dense(100,activation='relu'))
nn.add(Dense(10, activation='relu'))
nn.add(Dense(1, activation='linear'))
nn.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse', 'mae'])
nn.fit(X_dum_train, y_train, batch_size=32, epochs=20, verbose=1)

nn_train_pred = nn.predict(X_dum_train)
nn_valid_pred = nn.predict(X_dum_valid)
nn_r2_train=r2_score(y_train,nn_train_pred)
nn_mse_train = mean_squared_error(y_train,nn_train_pred)
nn_r2_valid = r2_score(y_valid,nn_valid_pred)
nn_mse_valid = mean_squared_error(y_valid, nn_valid_pred)
print('r^2 on train data:%f' % nn_r2_train,'MSE on train data:%f' % nn_mse_train)
print('r^2 on validation data:%f' % nn_r2_valid,'MSE on validation data:%f' %nn_mse_valid)
############################
#   Model Compartion       #
############################
method1=[0,1,2,3,4]
r2_train = [lasso_r2_train,rg_r2_train,cart_r2_train,rf_r2_train,nn_r2_train]
r2_valid= [lasso_r2_valid,rg_r2_valid,cart_r2_valid,rf_r2_valid,nn_r2_valid]
mse_train=[lasso_mse_train,rg_mse_train,cart_mse_train,rf_mse_train,nn_mse_train]
mse_valid=[lasso_mse_valid,rg_mse_valid,cart_mse_valid,rf_mse_valid,nn_mse_valid]

#comparison plot
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(method1,r2_train,color="red",linewidth=1,label="Training data")
plt.plot(method1,r2_valid,'b--',linewidth=1, label='Validation data')
plt.xlabel("Method")
plt.ylabel("R2")
plt.xticks(method1, ('Lasso', 'Ridge', 'CART', 'RF', 'NN'))

plt.subplot(1,2,2)
plt.plot(method1,mse_train, color="red",linewidth=1,label="Training data")
plt.plot(method1,mse_valid,'b--',linewidth=1, label='Validation data')
plt.xlabel("Method")
plt.ylabel("MSE")
plt.xticks(method1, ('Lasso', 'Ridge', 'CART', 'RF', 'NN'))
plt.legend(bbox_to_anchor=(1.05,1.0))

plt.savefig("r2comparison.png",bbox_inches='tight')

############################
#        Prediction        #
############################
#choose neural network model

#create dummy variables
predictor_catego = list(test_feature)[1:6]
predictor_conti = list(test_feature)[6:8]
dummy = pd.get_dummies(test_feature[predictor_catego])

X_dum_test = pd.concat([dummy,test_feature[predictor_conti]],axis=1)
nn_test_pred = nn.predict(X_dum_test)
jobId = test_feature.jobId

test_pred = pd.DataFrame({'jobId': jobId , 'salary': nn_test_pred[:,0]})
test_pred.to_csv('test_salaries.csv',columns=['jobId','salary'],index=False,sep=',')




