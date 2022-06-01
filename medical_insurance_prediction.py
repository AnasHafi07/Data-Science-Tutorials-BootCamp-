# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:11:51 2022

@author: ANAS
"""

#Step 1) Data Loading
#Step 2) Data Inspection
#Step 3) Data Cleaning
#Step 4) Features Selection
#Step 5) Pre-Processing  

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #label encoder for ml, another one when dl

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

#to visualize data #in inspection
#SMOTE for imbalanced classification


#for Categorical data
#to see the number of male and female in the dataset
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

# import matplotlib.pyplot as plt
# df.boxplot() # boxplot - explains the dispersion of the data

# #From the boxplot we can see that there are many outliers in charges

# import missingno as msno

# msno.matrix(df) #to visualized the NaNs of data
# msno.bar(df) #to visualized the NaNs of data

# #Here we can confirm that there are no NaNs for this datasets

# #So no need to impute
 

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

#if smart do a for loop
# category = ['sex','smoker', 'region'] 

# for i in category:
#     lr = LogisticRegression()
#     lr.fit(np.expand_dims(df['charges'],axis=-1),df['i'])   
#     print(i+str(lr.score(np.expand_dims(df['charges'],axis=-1),df['i']))

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



#select feature
#from the above analysis 
#drop children,sex,region
#keep age,bmi,smoker

#if you do regularization you'll get quite the same

#temp = np.array([1,2,3,4,5]).reshape(-1,1)
#stick to one only to reshape

X = df.loc[:,['age','bmi','smoker']]#features
y = df['charges']

# #To ensure the data for model training will only contain essential
# #Convince why that features. Sir said only 2 can be used. Sir buat

# #Fareez try lasso. MinMax one array. Expand only once

# #Regression analysis (Categorical vs Continuous data) #Logistic Regression

# #Point Biserial (Continuous Categorical)

# #categorical separate from continuous do one by one

# import seaborn as sns
# import matplotlib.pyplot as plt

# # #Method 1) Features selection using correlation
# cor = df.corr()

# plt.figure(figsize=(12,10))
# sns.heatmap(cor, annot=True,cmap =plt.cm.Reds)
# plt.show()

# #From the graph , smoker and charges shows strong correlation
# # only smoker and charges will be selected ML/DL training


# #Try Lasso
#Lasso boleh dua2 categorical and continous.

# from sklearn.preprocessing import StandardScaler
# X = df.drop(labels='charges', axis=1) 
# y = df['charges']

# scaler = StandardScaler()
# scaler.fit(X)

# from sklearn.linear_model import Lasso

# lasso = Lasso()

# lasso.fit(scaler.transform(X),y)

# lasso_coef = lasso.coef_

# print(lasso_coef)

# column_names = X.columns
# plt.plot(column_names, abs(lasso_coef))

# # plt.figure(figsize=(12,10))
# # plt.plot(column_names[0:-1],abs(lasso_coef))
# # plt.grid()

#%%
#Step 5) Data-preprocessing

#MIN MAX SCALLING
from sklearn.preprocessing import MinMaxScaler
import numpy as np
y=np.expand_dims(y,axis=-1)
mms = MinMaxScaler()
x_scaled = mms.fit_transform(X)
y_scaled = mms.fit_transform(y)

# # #Standardization
# from sklearn.preprocessing import StandardScaler
# std = StandardScaler()

# x_stand = std.fit_transform(X)
# y_stand = std.fit_transform(y)

#after this sir ajar which one scale to choose

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.3, random_state=123)
#X_train, X_test, y_train, y_test = train_test_split(x_stand, y_stand, test_size = 0.3, random_state=123)

#based train based on scale

#%%
# #Machine Learning 
# #Regression classification

# #%% machine learning 

# #Linear regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)

print(reg.score(X_test,y_test)) #r square value 1 is good 0 is bad

#accuracy not same as r

#%% Machine Learning Performance Evaluation

#which one is better ridge Lasso or linear