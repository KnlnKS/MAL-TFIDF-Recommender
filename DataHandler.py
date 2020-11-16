import pandas as pd
pd.set_option('display.max_columns', None)

class DataHandler():
    def __init__(self, is_limited):
        self._load_data(is_limited)
        self._remove_nans_from_genres()


    def _remove_nans(self, s):
        if isinstance(s, str):
            return s.split(',')
        else:
            return 'none'

    def _load_data(self, is_limited):
        self.anime = pd.read_csv('./input/anime_cleaned.csv')[['anime_id', 'title', 'score', 'genre']].sort_values(by='score', ascending=False)
        if not is_limited:
            self.anime_lists = pd.read_csv('./input/animelists_cleaned.csv')[
                ['username', 'anime_id', 'my_score']]
        else:
            self.anime_lists = pd.read_csv('./input/animelists_cleaned.csv', nrows=1000000)[
                ['username', 'anime_id', 'my_score']]


    def _remove_nans_from_genres(self):
        self.anime['genre'] = self.anime['genre'].apply(self._remove_nans)



    def get_anime_info(self, anime_id):
        return self.anime[self.anime['anime_id'] == anime_id]

    def get_anime_head(self):
        return self.anime.head()

