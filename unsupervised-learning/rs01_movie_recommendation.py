
"""
RS01 - Movie Recommendation System

Objective:
Explore movie ratings and build a basic user-based
collaborative filtering recommendation system.
"""


from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Load the datasets
# ============================================================

DATASET_PATH = (
    Path(__file__).parent
    / "datasets"
    / "ml-latest-small"
)

ratings = pd.read_csv(DATASET_PATH / "ratings.csv")
movies = pd.read_csv(DATASET_PATH / "movies.csv")

print("=== Dataset Overview ===")
print(f"Number of ratings: {len(ratings)}")
print(f"Number of movies: {len(movies)}")
print(f"Number of users: {ratings['userId'].nunique()}")

print("\nRatings:")
print(ratings.head())

print("\nMovies:")
print(movies.head())


# ============================================================
# 2. Explore the ratings
# ============================================================

print("\n=== Ratings Statistics ===")
print(ratings["rating"].describe())

print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index())

print(f"\nAverage rating: {ratings['rating'].mean():.2f}")
print(f"Unique users: {ratings['userId'].nunique()}")
print(f"Unique rated movies: {ratings['movieId'].nunique()}")


# ============================================================
# 3. Find the most popular movies
# ============================================================

popular_movies = (
    ratings.groupby("movieId")
    .size()
    .reset_index(name="rating_count")
    .merge(movies, on="movieId", how="left")
    .sort_values("rating_count", ascending=False)
)

print("\n=== Top 10 Most Popular Movies ===")
print(
    popular_movies[
        ["title", "genres", "rating_count"]
    ].head(10).to_string(index=False)
)


# ============================================================
# 4. Find the highest-rated movies
# ============================================================

MIN_RATINGS = 50

movie_statistics = (
    ratings.groupby("movieId")["rating"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(
        columns={
            "count": "rating_count",
            "mean": "average_rating",
        }
    )
)

highest_rated_movies = (
    movie_statistics[
        movie_statistics["rating_count"] >= MIN_RATINGS
    ]
    .merge(movies, on="movieId", how="left")
    .sort_values(
        ["average_rating", "rating_count"],
        ascending=[False, False],
    )
)

print(f"\n=== Top 10 Highest-Rated Movies (minimum {MIN_RATINGS} ratings) ===")
print(
    highest_rated_movies[
        ["title", "genres", "average_rating", "rating_count"]
    ].head(10).to_string(index=False)
)


# ============================================================
# 5. Create the user-item matrix
# ============================================================

user_item_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating",
)

total_possible_ratings = (
    user_item_matrix.shape[0] * user_item_matrix.shape[1]
)

existing_ratings = int(user_item_matrix.notna().sum().sum())
missing_ratings = total_possible_ratings - existing_ratings

sparsity = (
    missing_ratings / total_possible_ratings * 100
    if total_possible_ratings > 0
    else 0
)

print("\n=== User-Item Matrix ===")
print(f"Matrix shape: {user_item_matrix.shape}")
print(f"Possible ratings: {total_possible_ratings:,}")
print(f"Existing ratings: {existing_ratings:,}")
print(f"Missing ratings: {missing_ratings:,}")
print(f"Sparsity: {sparsity:.2f}%")

print("\nFirst 5 users and first 10 movies:")
print(user_item_matrix.iloc[:5, :10])


# ============================================================
# 6. Compare two users
# ============================================================

user_1_id = 1
user_2_id = 5

if user_1_id in user_item_matrix.index and user_2_id in user_item_matrix.index:
    user_1_ratings = user_item_matrix.loc[user_1_id]
    user_2_ratings = user_item_matrix.loc[user_2_id]

    common_movies_mask = (
        user_1_ratings.notna() & user_2_ratings.notna()
    )

    common_movies = pd.DataFrame({
        f"user_{user_1_id}": user_1_ratings[common_movies_mask],
        f"user_{user_2_id}": user_2_ratings[common_movies_mask],
    })

    print(f"\n=== Compare Users {user_1_id} and {user_2_id} ===")
    print(f"User {user_1_id} ratings: {user_1_ratings.notna().sum()}")
    print(f"User {user_2_id} ratings: {user_2_ratings.notna().sum()}")
    print(f"Common movies: {len(common_movies)}")
    print(common_movies)


# ============================================================
# 7. Calculate cosine similarity between two users
# ============================================================

if common_movies_mask.sum() > 0:
    similarity = cosine_similarity(
        user_1_ratings[common_movies_mask].to_numpy().reshape(1, -1),
        user_2_ratings[common_movies_mask].to_numpy().reshape(1, -1),
    )[0, 0]

    print(
        f"\nCosine similarity between users "
        f"{user_1_id} and {user_2_id}: {similarity:.4f}"
    )
else:
    print("\nThe selected users have no movies in common.")


# ============================================================
# 8. Define the recommendation function
# ============================================================

def recommend_movies(
    user_id: int,
    n_recommendations: int = 10,
    n_neighbors: int = 5,
    min_common_movies: int = 10,
    min_neighbor_votes: int = 2,
) -> pd.DataFrame:
    """
    Recommend movies to a user based on similar users' ratings.

    Args:
        user_id: ID of the user receiving recommendations.
        n_recommendations: Maximum number of movies to recommend.
        n_neighbors: Number of similar users to consider.
        min_common_movies: Minimum shared movies for user similarity.
        min_neighbor_votes: Minimum neighbor ratings required per movie.

    Returns:
        A DataFrame containing recommended movies and predicted ratings.
    """

    if user_id not in user_item_matrix.index:
        raise ValueError(
            f"User ID {user_id} was not found in the dataset."
        )

    if n_recommendations < 1:
        raise ValueError("n_recommendations must be at least 1.")

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    if min_common_movies < 1:
        raise ValueError("min_common_movies must be at least 1.")

    if min_neighbor_votes < 1:
        raise ValueError("min_neighbor_votes must be at least 1.")

    target_ratings = user_item_matrix.loc[user_id]

    # Find users with similar movie preferences.
    similar_users = []

    for other_user_id in user_item_matrix.index:
        if other_user_id == user_id:
            continue

        other_ratings = user_item_matrix.loc[other_user_id]

        common_mask = (
            target_ratings.notna() & other_ratings.notna()
        )

        common_count = int(common_mask.sum())

        if common_count < min_common_movies:
            continue

        target_common_ratings = (
            target_ratings[common_mask].to_numpy().reshape(1, -1)
        )

        other_common_ratings = (
            other_ratings[common_mask].to_numpy().reshape(1, -1)
        )

        similarity = cosine_similarity(
            target_common_ratings,
            other_common_ratings,
        )[0, 0]

        if similarity > 0:
            similar_users.append({
                "userId": other_user_id,
                "similarity": similarity,
                "common_movies": common_count,
            })

    if not similar_users:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "genres",
                "predicted_rating",
                "neighbor_count",
            ]
        )

    similar_users_df = (
        pd.DataFrame(similar_users)
        .sort_values(
            by=["similarity", "common_movies"],
            ascending=[False, False],
        )
        .head(n_neighbors)
    )

    # Identify movies that the target user has not rated.
    unrated_movies = target_ratings[
        target_ratings.isna()
    ].index

    recommendations = []

    for movie_id in unrated_movies:
        weighted_sum = 0.0
        similarity_sum = 0.0
        neighbor_count = 0

        for neighbor in similar_users_df.itertuples(index=False):
            neighbor_rating = user_item_matrix.at[
                neighbor.userId,
                movie_id,
            ]

            if pd.isna(neighbor_rating):
                continue

            weighted_sum += neighbor.similarity * neighbor_rating
            similarity_sum += neighbor.similarity
            neighbor_count += 1

        # Avoid recommendations supported by too few ratings.
        if neighbor_count < min_neighbor_votes:
            continue

        if similarity_sum == 0:
            continue

        predicted_rating = weighted_sum / similarity_sum

        recommendations.append({
            "movieId": movie_id,
            "predicted_rating": predicted_rating,
            "neighbor_count": neighbor_count,
        })

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "genres",
                "predicted_rating",
                "neighbor_count",
            ]
        )

    recommendations_df = pd.DataFrame(recommendations)

    # Add movie titles and genres.
    recommendations_df = recommendations_df.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left",
    )

    # Rank movies by predicted rating and supporting neighbor count.
    recommendations_df = recommendations_df.sort_values(
        by=["predicted_rating", "neighbor_count"],
        ascending=[False, False],
    )

    return recommendations_df[
        [
            "movieId",
            "title",
            "genres",
            "predicted_rating",
            "neighbor_count",
        ]
    ].head(n_recommendations).reset_index(drop=True)


# ============================================================
# 9. Generate recommendations for a selected user
# ============================================================

target_user_id = 25

try:
    recommended_movies = recommend_movies(
        user_id=target_user_id,
        n_recommendations=10,
        n_neighbors=5,
    )

    print(f"\n=== Movie Recommendations for User {target_user_id} ===")

    if recommended_movies.empty:
        print("No recommendations were found for this user.")
    else:
        print(recommended_movies.to_string(index=False))

except ValueError as error:
    print(f"\nRecommendation error: {error}")


# ============================================================
# 10. Project summary
# ============================================================

print("\n=== Project Summary ===")
print("Dataset: MovieLens latest-small")
print("Approach: User-based collaborative filtering")
print("Similarity metric: Cosine similarity")
print("Recommendation score: Similarity-weighted average rating")
