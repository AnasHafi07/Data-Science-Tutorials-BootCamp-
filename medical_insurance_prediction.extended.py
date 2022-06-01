# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:07:57 2022

@author: ANAS
"""

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #label encoder for ml, another one when dl
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

#amik raw jika nak download
PATH = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(PATH)

#%%STEP 2) Data Interpretation / Data Inspection

df.head()
df.tail()


df.info() # to check if there is any NaN/ to have a detailed info abt

# From the above steps we can conclude that there is no NaN 
df.describe().T
stats=df.describe().T
df.columns
df.isna().sum
df.duplicated().sum() #check for duplicated data
df[df.duplicated()] #to exteract the duplicated data

plt.figure()
sns.countplot(df['sex'])
plt.show

#smoker
plt.figure()
sns.countplot(df['smoker'])
plt.show

#region
plt.figure()
sns.countplot(df['region'])
plt.show

#for continous data
#age
plt.figure()
sns.distplot(df['age'])
plt.show()
#not so nice

#bmi
plt.figure()
sns.distplot(df['bmi'])
plt.show()
#this one ideal

#children
plt.figure()
sns.distplot(df['children'])
plt.show()

#charges
plt.figure()
sns.distplot(df['charges'])
plt.show()

#%% STEP 3) DATA CLEANING

df = df.drop_duplicates() #remove duplicates
df.duplicated().sum()
df[df.duplicated()] # see nothing inside
#No NaNs to remove/impute

#%% Step 4) features selection

#categorical features : Sex, smoker, region
#continous features : age, bmi, children,charges

#label encoding 

le = LabelEncoder()
df['sex']=le.fit_transform(df['sex']) #now 0 - female, 1 - male
print(df['sex'].unique())

print(le.inverse_transform(df['sex'].unique()))

df['smoker']=le.fit_transform(df['smoker']) #now 1-yes, 0 - no

print(df['smoker'].unique())

print(le.inverse_transform(df['smoker'].unique()))

df['region']=le.fit_transform(df['region'])
 #now [3 2 1 0] ['southwest' 'southeast' 'northwest' 'northeast']

print(df['region'].unique())

print(le.inverse_transform(df['region'].unique()))

#Regression analysis
#continous data
continous_data = df.loc[:,['age', 'bmi','children','charges']]
continous_data.corr()

plt.figure(figsize=(12,10))
sns.heatmap(continous_data.corr(), annot=True,cmap =plt.cm.Blues)
plt.show()
#check weak or not in slides
#although encode categorical still cannot use pearson corr

#categorical vs continous data
from sklearn.linear_model import LogisticRegression
import numpy as np

lr = LogisticRegression()
lr.fit(np.expand_dims(df['charges'],axis=-1),df['sex']) 
# # x need to be continous data, y is categorical data
lr.score(np.expand_dims(df['charges'],axis=-1),df['sex']) 
# lr.score shows the accuracy only
# 0.5 it shows not much correclation at all charges and sex

lr.fit(np.expand_dims(df['charges'],axis=-1),df['smoker']) 
# # x need to be continous data, y is categorical data
lr.score(np.expand_dims(df['charges'],axis=-1),df['smoker']) 

lr.fit(np.expand_dims(df['charges'],axis=-1),df['region']) 
# # x need to be continous data, y is categorical data
lr.score(np.expand_dims(df['charges'],axis=-1),df['region']) 


X = df.loc[:,['age','bmi','smoker']]#features
y = df['charges']

#%%
#Step 5) Data-preprocessing


#MIN MAX SCALLING
from sklearn.preprocessing import MinMaxScaler
import numpy as np
y=np.expand_dims(y,axis=-1)

# mms = MinMaxScaler()
# x_scaled = mms.fit_transform(X)
# y_scaled = mms.fit_transform(y)

# # #Standardization
from sklearn.preprocessing import StandardScaler
# std = StandardScaler()

# x_stand = std.fit_transform(X)
# y_stand = std.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=123)

# Steps for Standard Scaler Ridge
step_ss_ridge = [('Standard Scaler', StandardScaler()),
           ('Ridge Classifier', Ridge())] # Standard Scaling

# Steps for Min max Scaler Ridge
step_mms_ridge = [('Min Max Scaler', MinMaxScaler()),
             ('Ridge Classifier', Ridge())]

# Steps for Standard Scaler LR
step_ss_lr = [('Standard Scaler', StandardScaler()),
           ('Linear Classifier', LinearRegression())] # Standard Scaling

# Steps for Min max Scaler LR
step_mms_lr = [('Min Max Scaler', MinMaxScaler()),
             ('Linear Classifier', LinearRegression())]

# Steps for Standard Scaler Lasso
step_ss_ls = [('Standard Scaler', StandardScaler()),
           ('Lasso Classifier', Lasso())] # Standard Scaling

# Steps for Min max Scaler Lasso
step_mms_ls = [('Min Max Scaler', MinMaxScaler()),
             ('Lasso Classifier', Lasso())]



pipeline_ss_ridge = Pipeline(step_ss_ridge)

pipeline_mms_ridge = Pipeline(step_mms_ridge)

pipeline_ss_lr = Pipeline(step_ss_lr)

pipeline_mms_lr = Pipeline(step_mms_lr)

pipeline_ss_ls = Pipeline(step_ss_ls)

pipeline_mms_ls = Pipeline(step_mms_ls)

pipelines = [pipeline_ss_ridge, pipeline_mms_ridge, 
             pipeline_ss_lr, pipeline_mms_lr, 
             pipeline_ss_ls, pipeline_mms_ls]

#fitting of data
for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
pipe_dict = {0: 'Standard Scaler Approach R', 1: 'Min-Max Scaler Approach R',
             2: 'Standard Scaler Approach LR', 3: 'Min-Max Scaler Approach LR',
             4: 'Standard Scaler Approach LS', 5: 'Min-Max Scaler Approach LS'} 

best_accuracy = 0    
#model evaluation
for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]
        
print('The best scaling approach for IRIS Dataset will be {} with accuracy {}'.format(best_scaler, best_accuracy))

#%% Machine Learning Performance Evaluation

#which one is better ridge, Lasso or linear

#train test split


#X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.3, random_state=123)
#X_train, X_test, y_train, y_test = train_test_split(x_stand, y_stand, test_size = 0.3, random_state=123)

