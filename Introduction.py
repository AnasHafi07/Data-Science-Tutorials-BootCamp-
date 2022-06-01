# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:04:18 2022

This scrip is to introduce the concept of tensors

@author: ANAS
"""

import numpy as np

# Rank 0
rank_0 = 123 #scalar
np.shape(rank_0)

#%% Rank 1
rank_1 = [1,2,3] #vector
np.shape(rank_1)

# Rank 2
rank_2 = [[1,2,3], [2,3,4]]
np.shape(rank_2)

rank_2_arr = np.array(rank_2) #to convert into array

#%%
# Rank 3

# Method 1 (Easiest Method / Straighforward)
print('before' + str(np.shape(rank_1)))
method_1 = np.expand_dims(rank_1, axis = -1) #see when you change axis -1 and 0
print('after' + str(np.shape(method_1)))

#Method 2 (Pythonic)
method_2 = np.array(rank_1)
method_2 = method_2[None,:] # can higher rank by adding none
print(np.shape(method_2)) 

#%% Method 3
method_3 = np.array(rank_1)
method_3 = method_3[:,np.newaxis]
print(np.shape(method_3))

#%% Method 4
method_4 = np.array(rank_1)
method_4 = np.reshape(method_4, (3,1,1))
print(np.shape(method_4))

#%% Examples

#fiction
book_1 = [20,10,100]
book_2 = [20,10,50]
book_3 = [20,10,12] #Height, Width, Pages

#Fashion
book_4 = [20,10,120]
book_5 = [10,10,6]
book_6 = [25,20,3] #Height, Width, Pages

#Books ==>[Fiction, Fashion]
books = [[book_1,book_2,book_3], [book_4,book_5,book_6]]
print(np.shape(np.array(books)))

book_1_arr = np.array(book_1)

#%% 

#create tensors using tensorflow
import tensorflow as tf

#Tensor 0 (Magnitude)
scalar = tf.Variable(4,tf.int16) #rank0
print(scalar)

#Tensor 1 (Vector)
vector = tf.Variable([1,2,3,4], tf.int16)
print(vector)

#Tensor 2 (Matrix)
matrix = tf.Variable([[1,2,3,],[4,5,6]], tf.int16)
print(matrix)

#Tensor 3 (Rank 3)
tensor_3 = tf.Variable([[[1,2,3],[4,5,6]]],tf.int16)
print(tensor_3)

    