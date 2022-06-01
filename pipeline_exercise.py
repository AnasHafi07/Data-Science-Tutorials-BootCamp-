# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:39:21 2022

@author: ANAS
"""

# EDA
# Step 1) Data Loading
# Step 2) Data Inspection / visualization
# Step 3) Data Cleaning
# Step 4) Features selection
# Step 5) Preprocessing
# Machine Learning Model training

from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#%% Step 1) Data Loading
iris = datasets.load_iris()

print(iris.DESCR) 
# Sepal Length
# Sepal Width
# Petal Length
# Petal Width
# 3 Classes 0,1,2 --> Sentosa, Versicolor, Verginica
print(dir(iris)) #directory

X = iris.data
y = iris.target
print(iris.target_names)

#X.info
#Will not work since array not dataframe


X = pd.DataFrame(X) #To convert array into DF
X.columns = ['Sepal Length','Sepal Width', 'Petal Length', 'Petal Width'] 

X.info() #to check data type and NaNs
X.duplicated().sum() #GOT 1
X[X.duplicated()]

#to visualize the no categories
plt.figure()
sns.countplot(y)
plt.show() #even

#to visualize continous data
for i in X.columns:
    
    plt.figure()
    sns.distplot(X[i])
    plt.show()
    
X.boxplot()

X.describe().T


#%% Step 3) Data cleaning

#X.duplicates() #you may remove but if remove may gives imbalance in data 
#Not removing the duplicated to ensure the balance in numbers of data

#%% Step 4) Features Selection

# No need to do since this one already good data
# If want also can corr using logistic reg

#%% Step 5) Data preprocessing

#Pipeline

# We dont know which is best in terms of scaling

# To obtain training and testing data using train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=123)

#Pipeline creation
# Step 1) Apply features scalling appcorach(Minx-max & Std Scaling)



# Steps for Standard Scaler
step_ss = [('Standard Scaler', StandardScaler()),
           ('Logistic Classifier', LogisticRegression())] # Standard Scaling

# Steps for Min max Scaler
step_mms = [('Min Max Scaler', MinMaxScaler()),
             ('Logistic Classifier', LogisticRegression())]

#to create pipeline
pipeline_ss = Pipeline(step_ss)

pipeline_mms = Pipeline(step_mms)

#Create a list for the pipeline so that tyou can iterate them
pipelines = [pipeline_ss, pipeline_mms]

#fitting of data
for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
pipe_dict = {0: 'Standard Scaler Approach', 1: 'Min-Max Scaler Approach'}   
best_accuracy = 0    
#model evaluation
for i, model in enumerate(pipelines):
    #print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]
        
print('The best scaling approach for IRIS Dataset will be {} with accuracy {}'.format(best_scaler, best_accuracy))

#%% saving the optimal model
import pickle

pkl_fname = 'best_model.pkl'
with open(pkl_fname,'wb') as file:
    pickle.dump(best_pipeline,file)
    
#%%    
#to load the best pipeline 
with open (pkl_fname,'rb') as file:
    pickle_pipeline = pickle.load(file)
    
pickle_pipeline.score(X_test,y_test)

# #%%
# pipeline1 =
# pipeline2 = 
# pipelines = pp1,pp2