import pandas as pd
pd.set_option('display.max_columns', None)

class DataHandler():
    def __init__(self, is_limited):
        self._load_data(is_limited)
        self._generate_genre_ids()

    def _string_to_list(self, s):
        if isinstance(s, str):
            return s.split(',')
        else:
            return 'none'

    def _list_to_string(self, a):
            return str(a)[1:-1]

    def _load_data(self, is_limited):
        self.anime = pd.read_csv('./input/anime_cleaned.csv')[['anime_id', 'title', 'score', 'rank', 'genre']]
        if not is_limited:
            self.anime_lists = pd.read_csv('./input/animelists_cleaned.csv')[
                ['username', 'anime_id', 'my_score']]
        else:
            self.anime_lists = pd.read_csv('./input/animelists_cleaned.csv', nrows=1000000)[
                ['username', 'anime_id', 'my_score']]


    def _generate_genre_ids(self):
        self.anime['genre'] = self.anime['genre'].apply(self._string_to_list)
        v = self.anime['genre'].explode()
        v[:] = pd.factorize(v)[0] + 1
        self.anime['genre'] = v.groupby(level=0).agg(list)
        self.anime['genre'] = self.anime['genre'].apply(self._list_to_string)




    def get_anime_info(self, anime_id):
        return self.anime[self.anime['anime_id'] == anime_id]

    def get_anime_head(self):
        return self.anime.head()

