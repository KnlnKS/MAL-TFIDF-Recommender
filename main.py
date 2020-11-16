import time as t

import DataHandler as d
import TFIDF as tf
import UserProfiler as u


# Load in score data
start = t.time()
dataHandler = d.DataHandler(False)
print('Data loaded and preprocessed in ' + str(t.time() - start) + ' seconds.')
print()

similarities = tf.TFIDF(dataHandler.anime)
recommender = u.UserProfiler(similarities, dataHandler)
#print(recommender.get_user_ratings('Kvazikvark').sort_values(by='my_score', ascending=False))
m = recommender.get_user_recommendations('Kvazikvark')
print(m.size)
print(m.head())
