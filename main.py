"""
main.py

Simple command-line demo for the MovieRecommender.

Usage examples:
    python main.py --mode content --title "Toy Story"
    python main.py --mode collaborative --user 3
    python main.py --mode both --title "The Matrix" --user 3
    python main.py --list-movies
    python main.py --list-users
"""

import argparse

from recommender import MovieRecommender


def print_header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def run_content(recommender: MovieRecommender, title: str, n: int):
    print_header(f"Content-Based Recommendations (similar to '{title}')")
    try:
        results = recommender.content_based_recommend(title, n=n)
        for i, row in results.iterrows():
            print(f"{i + 1}. {row['title']}  [{row['genres']}]  "
                  f"similarity={row['similarity']}")
    except ValueError as e:
        print(f"Error: {e}")


def run_collaborative(recommender: MovieRecommender, user_id: int, n: int):
    print_header(f"Collaborative Filtering Recommendations (for user {user_id})")

    try:
        top_rated = recommender.get_user_top_rated(user_id, n=5)
        print(f"\nUser {user_id}'s top-rated movies (for context):")
        for i, row in top_rated.iterrows():
            print(f"  - {row['title']} (rated {row['rating']})")

        results = recommender.collaborative_recommend(user_id, n=n)
        print(f"\nRecommended movies for user {user_id}:")
        for i, row in results.iterrows():
            print(f"{i + 1}. {row['title']}  [{row['genres']}]  "
                  f"predicted_rating={row['predicted_rating']}")
    except ValueError as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument(
        "--mode",
        choices=["content", "collaborative", "both"],
        default="both",
        help="Which recommendation technique to run.",
    )
    parser.add_argument("--title", type=str, help="Movie title for content-based filtering.")
    parser.add_argument("--user", type=int, help="User ID for collaborative filtering.")
    parser.add_argument("--n", type=int, default=5, help="Number of recommendations to return.")
    parser.add_argument("--list-movies", action="store_true", help="List all movie titles.")
    parser.add_argument("--list-users", action="store_true", help="List all user IDs.")

    args = parser.parse_args()
    recommender = MovieRecommender()

    if args.list_movies:
        print_header("Available Movies")
        for title in recommender.list_movie_titles():
            print(f"  - {title}")
        return

    if args.list_users:
        print_header("Available Users")
        print(recommender.list_user_ids())
        return

    if args.mode in ("content", "both"):
        title = args.title or "Toy Story"
        run_content(recommender, title, args.n)

    if args.mode in ("collaborative", "both"):
        user_id = args.user or recommender.list_user_ids()[0]
        run_collaborative(recommender, user_id, args.n)


if __name__ == "__main__":
    main()
