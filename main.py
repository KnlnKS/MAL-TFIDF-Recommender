import time as t

import DataHandler as d
import TFIDF as tf
import UserProfiler as u


# Load in score data
start = t.time()
dataHandler = d.DataHandler(False)
print('Data loaded and preprocessed in ' + str(t.time() - start) + ' seconds.')
print()

# Use TFIDF algo
start = t.time()
similarities = tf.TFIDF(dataHandler.anime)
print('Genre importance calculated in ' + str(t.time() - start) + ' seconds.')
print()

# Get recommendations
start = t.time()
recommender = u.UserProfiler(similarities, dataHandler)
print('Recommednations generated in ' + str(t.time() - start) + ' seconds.')
print()

watched_anime = recommender.get_user_ratings('user in animelist_cleaned.csv').sort_values(by='my_score', ascending=False)
recommendations = recommender.get_user_recommendations('user in animelist_cleaned.csv')

print('User has watched '+str(watched_anime.size)+' anime.')
print('Fetched '+str(recommendations.size)+' recommendations.')
print()
print(recommendations.head())
