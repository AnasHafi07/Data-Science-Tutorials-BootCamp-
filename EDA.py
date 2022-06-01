# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:04:40 2022

This ETA Script is to perform EDA steps

Step 1) Data Loading

# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's

@author: ANAS
"""

import pandas as pd
import os 


#print(os.getcwd())
#print(os.path.join(os,getcwd(), 'housing.csv')) #cwd + housing.csv
DATA_PATH = os.path.join(os.getcwd(),'housing.csv') #something static all capital

#df = pd.read_csv(DATA_PATH)

#To comment one/multiple line(s) : hold ctrl +1
#df = pd.read_csv(DATA_PATH, sep=r's+') #Method 1 not recommended
#df = pd.read_csv(DATA_PATH, delim_whitespace=(True)) #Recommended method

#house = pd.read_csv('housing.csv') #hold shift right click to get the path

# current working directory
#house = pd.read_csv("E:\Studies\ANAS_SHRDC\Deep_Learning\housing.csv")
#print(house.head()) 

column_names = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', ' DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] # try not to cross

df = pd.read_csv(DATA_PATH, delim_whitespace=(True), names = column_names)

#recommended method
#df.columns = column_names

#%%STEP 2) Data Interpretation / Data Inspection

df.head()
df.tail()

df.info() # to check if there is any NaN/ to have a detailed info abt

#Dtype object - non-numeric data

df.describe().T # percentile, mean ,max, count 
#.T to transpose so that we can see everything

df.columns # to get column names

#visualize data 
#chas is categorical

#%% STEP 3) DATA CLEANING
import matplotlib.pyplot as plt
df.boxplot() # boxplot - explains the dispersion of the data

import missingno as msno

msno.matrix(df) #to visualized the NaNs of data
msno.bar(df) #to visualized the NaNs of data

df['CRIM'] = pd.to_numeric(df['CRIM'], errors = 'coerce') #if values covert to value else to NaN

df['ZN'] = pd.to_numeric(df['ZN'], errors = 'coerce') #if smart do at early in step3
#df_backup = df.copy()
df.info() # to check the oject converted to float

msno.matrix(df) #to visualized the Nans of data
msno.bar(df) #to visualized the nans of data

df.boxplot() #now more outliers because before this NaNs
df.describe().T

# Dealing with Nans
df['CRIM'].isna().sum() #to check the total of Nan #x sama sbb sir dapat 2
df['ZN'].isna().sum() #
df['MEDV'].isna().sum()

# if too many Nan you may want to drop the columns
# sir do not fancy dropping columns
#Dropping NANs least favourable apporach

#Method 1) Drop Nan
#try not to drop if possible

# Method 2) Impute NaNs
#df['CRIM']= df['CRIM'].fillna(df['CRIM'].median()) #dont use mean because prone to outliers
#df['ZN'] = df['ZN'].fillna(df['ZN'].median()) 

# df.info()
# df.describe().T

#Using Sklearn library to impute
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df = imputer.fit_transform(df)
df = pd.DataFrame(df) #to change back since sklearn convert to array
df.columns = column_names

#KKN imputer
# from sklearn.impute import KNNImputer

# knn_imputer = KNNImputer(n_neighbors = 5, metric='nan_euclidean')
# #usually n we took the sqrt of n but need to be odd number


# imputed_data = knn_imputer.fit_transform(df)
# df = pd.DataFrame(imputed_data)
# df.columns = column_names
# print(df.describe().T)

# #MICE
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer



# ii_imputer = IterativeImputer()
# imputed_data_ii = ii_imputer.fit_transform(df_backup)
# df_ii = pd.DataFrame(imputed_data_ii)
# df_ii.columns=column_names
# print(df_ii.describe().T)
# #this one not good since ade negative value
# #this one uses linear regression

#%% Step 4) features selection
#To ensure the data for model training will only contain essential
import seaborn as sns
import matplotlib.pyplot as plt

# #Method 1) Features selection using correlation
cor = df.corr()

plt.figure(figsize=(12,10))
sns.heatmap(cor, annot=True,cmap =plt.cm.Reds)
plt.show()

#From the graph , RM and LSTAT shows strong correlation
# only RM and LSTAT will be selected ML/DL training
#Pearson correlation useful only for continuous data

#So when categorical use cramers v

#Method 2) Features selection using lasso regression

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import Lasso

#to inisiate Lasso method
lasso = Lasso()

X = df.drop(labels = ['MEDV'], axis=1) #Features
y = df['MEDV'] #Target

lasso.fit(X,y)
lasso_coef = lasso.coef_
print(lasso_coef)

plt.figure(figsize=(12,10))
plt.plot(column_names[0:-1],abs(lasso_coef))
plt.grid()

#%%
#Step 5) Data-preprocessing
#RM, PTRATIO, LSTAT

X = df.loc[:,['RM','PTRATIO','LSTAT']] #error 
y = df['MEDV']


# #MIN MAX SCALLING
from sklearn.preprocessing import MinMaxScaler
import numpy as np
y=np.expand_dims(y,axis=-1)
mms = MinMaxScaler()
x_scaled = mms.fit_transform(X)
y_scaled = mms.fit_transform(y)

# #Standardization
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

x_stand = std.fit_transform(X)
y_stand = std.fit_transform(y)

# #train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_stand, y_stand, test_size = 0.3, random_state=123)

#%% machine learning 

#Linear regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)

print(reg.score(X_test,y_test))

#medical insurance prediction
