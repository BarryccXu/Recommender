# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import graphlab
song_data = graphlab.SFrame('song_data.gl/')
song_data.head()

graphlab.canvas.set_target('browser')
song_data['song'].show()
len(song_data)
## count numbers of users
users = song_data['user_id'].unique()
len(users)
##create a song recommender
train_data, test_data = song_data.random_split(0.8, seed = 0)
## simple popularity-based recommender
popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id = 'user_id',
                                                         item_id = 'song')
## everyone gets the same recommandation
popularity_model.recommend(users = [users[0]])
popularity_model.recommend(users = [users[1]])
## recommender with personalization
personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                         user_id = 'user_id',
                                                         item_id = 'song')
personalized_model.recommend(users = [users[0]])
personalized_model.recommend(users = [users[1]])

personalized_model.get_similar_items(['With Or Without You - U2', 'The Cove - Jack Johnson'])
#quantitative comparison btw models
#import matplotlib
model_performance = graphlab.recommender.util.compare_models(test_data,
                                                            [popularity_model, personalized_model],
                                                            user_sample = 1)