# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:47:42 2017
source: http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2
https://github.com/ictar/python-doc/blob/master/Science%20and%20Data%20Analysis/
%E5%9C%A8Python%E4%B8%AD%E5%AE%9E%E7%8E%B0%E4%BD%A0%E8%87%AA%E5%B7%B1%E7%9A%84%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F.md
@author: BarryXU
"""

import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
   train_data_matrix[line[1]-1, line[2]-1] = line[3]  
    
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

test_item = test_data_matrix
test_item[test_item.nonzero()] = 1
    
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

## CF based on users/items
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #do normalization
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

#user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, item_similarity, type='item')

# elvaluation with MSE
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

#print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
    
prediction = item_prediction[test_data_matrix.nonzero()].flatten() 
ground_truth = test_data_matrix[test_data_matrix.nonzero()].flatten()




