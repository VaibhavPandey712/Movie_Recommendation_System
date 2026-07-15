"""
recommender.py

A small movie recommendation engine implementing two classic techniques:

1. Content-Based Filtering
   Recommends movies similar to a given movie by comparing genres using
   TF-IDF vectorization + cosine similarity.

2. Collaborative Filtering (User-Based)
   Recommends movies for a given user by finding similar users (based on
   rating patterns) and suggesting movies those similar users liked.

Both techniques work off two simple CSV files:
   data/movies.csv   -> movieId, title, genres (pipe-separated)
   data/ratings.csv  -> userId, movieId, rating
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


class MovieRecommender:
    def __init__(self, movies_path: str = None, ratings_path: str = None):
        movies_path = movies_path or DATA_DIR / "movies.csv"
        ratings_path = ratings_path or DATA_DIR / "ratings.csv"

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        self._build_content_similarity()
        self._build_user_item_matrix()

    # ------------------------------------------------------------------
    # Content-Based Filtering
    # ------------------------------------------------------------------
    def _build_content_similarity(self):
        """Build a TF-IDF matrix over genres and a cosine similarity matrix."""
        genre_text = self.movies["genres"].fillna("").str.replace("|", " ", regex=False)

        vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z0-9\-]+")
        tfidf_matrix = vectorizer.fit_transform(genre_text)

        self.content_sim = cosine_similarity(tfidf_matrix)
        self._title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )

    def content_based_recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        """Recommend movies similar to `title` based on genre overlap."""
        key = title.lower().strip()
        if key not in self._title_to_index:
            matches = self.movies[
                self.movies["title"].str.lower().str.contains(key, na=False)
            ]
            if matches.empty:
                raise ValueError(f"Movie '{title}' not found in dataset.")
            idx = matches.index[0]
        else:
            idx = self._title_to_index[key]

        sim_scores = list(enumerate(self.content_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        result_idx = [i for i, _ in sim_scores]
        result = self.movies.iloc[result_idx][["movieId", "title", "genres"]].copy()
        result["similarity"] = [round(score, 3) for _, score in sim_scores]
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Collaborative Filtering (User-Based)
    # ------------------------------------------------------------------
    def _build_user_item_matrix(self):
        """Pivot ratings into a users x movies matrix (0 = not rated)."""
        self.user_item_matrix = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

        self.user_sim = cosine_similarity(self.user_item_matrix)
        self.user_sim_df = pd.DataFrame(
            self.user_sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def collaborative_recommend(
        self, user_id: int, n: int = 5, k_neighbors: int = 5
    ) -> pd.DataFrame:
        """
        Recommend movies for `user_id` using user-based collaborative filtering.

        Steps:
          1. Find the k most similar users to `user_id`.
          2. Compute a weighted average rating (weighted by similarity) for
             every movie those neighbors have rated but `user_id` has not.
          3. Return the top-n highest scoring movies.
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in ratings dataset.")

        similar_users = (
            self.user_sim_df[user_id]
            .drop(index=user_id)
            .sort_values(ascending=False)
            .head(k_neighbors)
        )

        if similar_users.sum() == 0:
            raise ValueError(f"No similar users found for user {user_id}.")

        target_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = target_ratings[target_ratings == 0].index

        scores = {}
        for movie_id in unrated_movies:
            neighbor_ratings = self.user_item_matrix.loc[similar_users.index, movie_id]
            mask = neighbor_ratings > 0
            if mask.sum() == 0:
                continue
            weights = similar_users[mask]
            score = np.average(neighbor_ratings[mask], weights=weights)
            scores[movie_id] = score

        if not scores:
            raise ValueError(f"Not enough data to recommend movies for user {user_id}.")

        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        movie_ids, predicted_scores = zip(*top_movies)

        result = self.movies[self.movies["movieId"].isin(movie_ids)][
            ["movieId", "title", "genres"]
        ].copy()
        score_map = dict(zip(movie_ids, predicted_scores))
        result["predicted_rating"] = result["movieId"].map(score_map).round(2)
        result = result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_user_top_rated(self, user_id: int, n: int = 5) -> pd.DataFrame:
        """Show a user's own highest-rated movies (useful for context)."""
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        merged = user_ratings.merge(self.movies, on="movieId")
        merged = merged.sort_values("rating", ascending=False).head(n)
        return merged[["movieId", "title", "genres", "rating"]].reset_index(drop=True)

    def list_movie_titles(self):
        return self.movies["title"].tolist()

    def list_user_ids(self):
        return sorted(self.ratings["userId"].unique().tolist())
