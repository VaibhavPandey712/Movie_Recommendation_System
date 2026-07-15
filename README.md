# 🎬 Movie Recommendation System

A simple, dependency-light movie recommender built in pure Python. It implements
two classic recommendation techniques side by side:

- **Content-Based Filtering** — recommends movies similar to one you already like,
  based on genre overlap (TF-IDF + cosine similarity).
- **Collaborative Filtering** — recommends movies for a specific user by finding
  similar users and suggesting what they rated highly (user-based CF).

No external APIs, no accounts, no heavy frameworks — just `pandas` and
`scikit-learn` over two small CSV files.

## How it works

### Content-Based Filtering
Each movie's genre list (e.g. `Action|Sci-Fi|Thriller`) is turned into a TF-IDF
vector. Cosine similarity between vectors tells us which movies share the most
genre "DNA" with a given title.

### Collaborative Filtering (User-Based)
Ratings are pivoted into a user × movie matrix. Cosine similarity between rows
finds users with similar taste to the target user. Movies those neighbors rated
highly — but the target user hasn't seen — are recommended, weighted by how
similar each neighbor is.

## Project structure

```
movie-recommender/
├── data/
│   ├── movies.csv          # movieId, title, genres
│   └── ratings.csv         # userId, movieId, rating
├── recommender.py          # MovieRecommender class (core logic)
├── main.py                 # CLI demo
├── test_recommender.py     # unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <this-repo>
cd movie-recommender
pip install -r requirements.txt
```

## Usage

Run both recommendation types at once:

```bash
python main.py --mode both --title "The Matrix" --user 3
```

Content-based only:

```bash
python main.py --mode content --title "Toy Story" --n 5
```

Collaborative filtering only:

```bash
python main.py --mode collaborative --user 7 --n 5
```

Browse what's available:

```bash
python main.py --list-movies
python main.py --list-users
```

### Sample output

```
============================================================
Content-Based Recommendations (similar to 'The Matrix')
============================================================
1. Spider-Man (2002)  [Action|Adventure|Sci-Fi]  similarity=0.905
2. Terminator 2: Judgment Day (1991)  [Action|Sci-Fi|Thriller]  similarity=0.861
3. War of the Worlds (2005)  [Action|Adventure|Sci-Fi|Thriller]  similarity=0.798
...

============================================================
Collaborative Filtering Recommendations (for user 3)
============================================================

User 3's top-rated movies (for context):
  - Dances with Wolves (1990) (rated 5.0)
  - Fight Club (1999) (rated 5.0)
  ...

Recommended movies for user 3:
1. Ghost (1990)  [Drama|Romance|Fantasy]  predicted_rating=4.5
2. The Departed (2006)  [Crime|Drama|Thriller]  predicted_rating=4.49
...
```

## Using it in your own code

```python
from recommender import MovieRecommender

rec = MovieRecommender()

# Content-based
similar = rec.content_based_recommend("Toy Story", n=5)

# Collaborative filtering
suggestions = rec.collaborative_recommend(user_id=3, n=5)
```

## Using your own data

Swap in your own CSVs as long as they follow this schema:

- `movies.csv`: `movieId,title,genres` (genres pipe-separated, e.g. `Action|Comedy`)
- `ratings.csv`: `userId,movieId,rating`

```python
rec = MovieRecommender(movies_path="data/my_movies.csv", ratings_path="data/my_ratings.csv")
```

The included datasets are small and synthetic (50 movies, 25 users) so the
project runs instantly and is easy to read through — swap in a larger dataset
like [MovieLens](https://grouplens.org/datasets/movielens/) for more realistic
results.

## Running tests

```bash
python -m pytest test_recommender.py -v
```

## Possible extensions

- Hybrid filtering (blend content-based + collaborative scores)
- Matrix factorization (SVD) for collaborative filtering at scale
- A small Flask/Streamlit front end
- Incorporate movie descriptions/tags into content-based similarity, not just genres

## License

MIT — do whatever you'd like with this.
