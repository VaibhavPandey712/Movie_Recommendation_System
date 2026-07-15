

import pytest

from recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender():
    return MovieRecommender()


def test_movies_and_ratings_load(recommender):
    assert len(recommender.movies) > 0
    assert len(recommender.ratings) > 0


def test_content_based_recommend_returns_n_results(recommender):
    results = recommender.content_based_recommend("Toy Story", n=5)
    assert len(results) == 5
    assert "title" in results.columns
    assert "similarity" in results.columns


def test_content_based_recommend_excludes_query_movie(recommender):
    results = recommender.content_based_recommend("Toy Story (1995)", n=10)
    assert "Toy Story (1995)" not in results["title"].values


def test_content_based_recommend_invalid_title_raises(recommender):
    with pytest.raises(ValueError):
        recommender.content_based_recommend("Not A Real Movie Title XYZ", n=5)


def test_content_based_partial_title_match(recommender):
    results = recommender.content_based_recommend("matrix", n=3)
    assert len(results) == 3


def test_collaborative_recommend_returns_results(recommender):
    user_id = recommender.list_user_ids()[0]
    results = recommender.collaborative_recommend(user_id, n=5)
    assert len(results) <= 5
    assert "predicted_rating" in results.columns


def test_collaborative_recommend_invalid_user_raises(recommender):
    with pytest.raises(ValueError):
        recommender.collaborative_recommend(user_id=999999, n=5)


def test_collaborative_excludes_already_rated_movies(recommender):
    user_id = recommender.list_user_ids()[0]
    already_rated = set(
        recommender.ratings[recommender.ratings["userId"] == user_id]["movieId"]
    )
    results = recommender.collaborative_recommend(user_id, n=10)
    recommended_ids = set(results["movieId"])
    assert already_rated.isdisjoint(recommended_ids)


def test_get_user_top_rated(recommender):
    user_id = recommender.list_user_ids()[0]
    top = recommender.get_user_top_rated(user_id, n=3)
    assert len(top) <= 3
    ratings = top["rating"].tolist()
    assert ratings == sorted(ratings, reverse=True)
