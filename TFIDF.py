import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF():
    def __init__(self, anime):
        tf_idf = TfidfVectorizer()

        self.dataset = anime
        self.dataset_transformed = tf_idf.fit_transform(anime.genre)

        self.anime_similarities = cosine_similarity(self.dataset_transformed)
        self.tfidf_similarities = pd.DataFrame(cosine_similarity(self.dataset_transformed))

        self.tfidf_similarities.columns = [str(anime['anime_id'][int(col)]) for col in self.tfidf_similarities.columns]
        self.tfidf_similarities.index = [anime['anime_id'][idx] for idx in self.tfidf_similarities.index]

    def find_top_5_similar_anime(self, anime_id):
        index = self.dataset.index[self.dataset['anime_id'] == anime_id]
        similar_anime = self.tfidf_similarities.iloc[index].transpose().sort_values(by=anime_id, ascending=False)[1:6]
        return self.dataset[self.dataset['anime_id'].isin(similar_anime.index.values)].title