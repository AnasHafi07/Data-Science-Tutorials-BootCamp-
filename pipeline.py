# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:11:19 2022

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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Steps for Standard Scaler Ridge
step_ss_ridge = [('Standard Scaler', StandardScaler()),
           ('Ridge Classifier', Ridge())] # Standard Scaling

# Steps for Min max Scaler Ridge
step_mms_ridge = [('Min Max Scaler', MinMaxScaler()),
             ('Ridge Classifier', Ridge())]

# Steps for Standard Scaler Tree
step_ss_tree = [('Standard Scaler', StandardScaler()),
           ('Tree Classifier',  DecisionTreeClassifier())] # Standard Scaling

# Steps for Min max Scaler Tree
step_mms_tree = [('Min Max Scaler', MinMaxScaler()),
             ('Tree Classifier',  DecisionTreeClassifier())]

# Steps for Standard Scaler Random Forest
step_ss_rf = [('Standard Scaler', StandardScaler()),
           ('Random Forest Classifier', RandomForestClassifier())] # Standard Scaling

# Steps for Min max Scaler Random Forest
step_mms_rf = [('Min Max Scaler', MinMaxScaler()),
             ('Random Forest Classifier',RandomForestClassifier())]

# Steps for Standard Scaler LR
step_ss_lr = [('Standard Scaler', StandardScaler()),
           ('Linear Classifier', LinearRegression())] # Standard Scaling

# Steps for Min max Scaler LR
step_mms_lr = [('Min Max Scaler', MinMaxScaler()),
             ('Linear Classifier', LinearRegression())]


#to create pipeline
pipeline_ss = Pipeline(step_ss)

pipeline_mms = Pipeline(step_mms)

pipeline_ss_ridge = Pipeline(step_ss_ridge)

pipeline_mms_ridge = Pipeline(step_mms_ridge)

pipeline_ss_tree = Pipeline(step_ss_tree)

pipeline_mms_tree = Pipeline(step_mms_tree)

pipeline_ss_rf = Pipeline(step_ss_rf)

pipeline_mms_rf = Pipeline(step_mms_rf)

pipeline_ss_lr = Pipeline(step_ss_lr)

pipeline_mms_lr = Pipeline(step_mms_lr)


#Create a list for the pipeline so that tyou can iterate them
# #pipelines = [pipeline_ss, pipeline_mms, pipeline_ss_ridge, pipeline_mms_ridge, 
#              pipeline_ss_tree, pipeline_mms_tree, pipeline_ss_rf, 
#              pipeline_mms_rf, pipeline_ss_lr, pipeline_mms_lr]

pipelines = [pipeline_ss_ridge, pipeline_mms_ridge, 
             pipeline_ss_tree, pipeline_mms_tree, pipeline_ss_rf, 
             pipeline_mms_rf, pipeline_ss_lr, pipeline_mms_lr]

#fitting of data
for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
# pipe_dict = {0: 'Standard Scaler Approach', 1: 'Min-Max Scaler Approach',
#              2: 'Standard Scaler Approach R', 3: 'Min-Max Scaler Approach R',
#              4: 'Standard Scaler Approach T', 5: 'Min-Max Scaler Approach T',
#              6: 'Standard Scaler Approach RF', 7: 'Min-Max Scaler Approach RF',
#              8: 'Standard Scaler Approach LR', 9: 'Min-Max Scaler Approach LR',}   

pipe_dict = {0: 'Standard Scaler Approach R', 1: 'Min-Max Scaler Approach R',
             2: 'Standard Scaler Approach T', 3: 'Min-Max Scaler Approach T',
             4: 'Standard Scaler Approach RF', 5: 'Min-Max Scaler Approach RF',
             6: 'Standard Scaler Approach LR', 7: 'Min-Max Scaler Approach LR',} 

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