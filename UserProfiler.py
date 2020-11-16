import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_rows", None, "display.max_columns", None)


class UserProfiler():
    def __init__(self, tfidf, dataHandler):
        self.dataHandler = dataHandler
        self.tfidf = tfidf

    def get_user_ratings(self, username):
        user_ratings = self.dataHandler.anime_lists.loc[self.dataHandler.anime_lists['username'] == username]
        return self.dataHandler.anime.reset_index().merge(user_ratings, on='anime_id')[['title', 'my_score']]

    def get_user_recommendations(self, username):
        user_ratings = self.get_user_ratings(username)
        user_ratings['weight'] = user_ratings['my_score'] / 10.

        profile = np.dot(self.tfidf.dataset_transformed[user_ratings.index.values].toarray().T,
                         user_ratings['weight'].values)
        cos_sim = cosine_similarity(np.atleast_2d(profile), self.tfidf.dataset_transformed)

        recs = np.argsort(cos_sim)[:, ::-1]
        recommendations = [i for i in recs[0] if i not in user_ratings.index.values]

        titles = self.dataHandler.anime['title'][recommendations]
        rank = self.dataHandler.anime['rank'][recommendations]
        final = ((pd.concat([titles, rank], axis=1)).sort_values(by='rank', ascending=True)).dropna()

        return final
