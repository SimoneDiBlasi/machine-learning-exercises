"""
Exercise RS03 - Hybrid Recommendation System

Objective:
    Build a movie recommendation system that combines
    collaborative filtering and content-based filtering.

Dataset:
    MovieLens latest-small

Concepts:
    - Collaborative Filtering
    - Content-Based Filtering
    - Multi-Hot Encoding
    - Cosine Similarity
    - Score Normalization
    - Weighted Hybrid Recommendations
    - Recommendation Ranking

Tasks:
    1. Load and explore the movie and rating datasets.
    2. Convert movie genres into numerical feature vectors.
    3. Calculate movie-to-movie content similarity.
    4. Calculate user-to-user collaborative similarity.
    5. Generate content-based and collaborative scores for a user.
    6. Combine normalized scores into hybrid recommendations.
    7. Compare recommendations from the individual and hybrid methods.

Expected Outcome:
    Given a user ID, return a ranked list of movies that combines
    content similarity and preferences from similar users.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


# ---------------------------------------------------------
# Step 1 - Load and explore the datasets
# ---------------------------------------------------------

DATASET_DIR = Path(__file__).parent / "datasets" / "ml-latest-small"

movies = pd.read_csv(DATASET_DIR / "movies.csv")
ratings = pd.read_csv(DATASET_DIR / "ratings.csv")

print("Movies dataset")
print(f"Number of movies: {len(movies)}")
print(movies.head())

print("\nRatings dataset")
print(f"Number of ratings: {len(ratings)}")
print(f"Number of users: {ratings['userId'].nunique()}")
print(f"Number of rated movies: {ratings['movieId'].nunique()}")
print(ratings.head())


# ---------------------------------------------------------
# Step 2 - Convert movie genres into numerical features
# ---------------------------------------------------------

movies["genre_list"] = movies["genres"].apply(
    lambda genres: (
        []
        if genres == "(no genres listed)"
        else genres.split("|")
    )
)

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies["genre_list"])

genre_features = pd.DataFrame(
    genre_matrix,
    columns=mlb.classes_,
    index=movies["movieId"],
)

print("\nAvailable genres")
print(list(mlb.classes_))
print(f"Genre feature matrix shape: {genre_features.shape}")


# ---------------------------------------------------------
# Step 3 - Calculate content-based movie similarity
# ---------------------------------------------------------

content_similarity_matrix = cosine_similarity(genre_features)

content_similarity = pd.DataFrame(
    content_similarity_matrix,
    index=genre_features.index,
    columns=genre_features.index,
)

print("\nContent similarity matrix shape:")
print(content_similarity.shape)


# ---------------------------------------------------------
# Step 4 - Create the user-item ratings matrix
# ---------------------------------------------------------

user_item_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating",
).fillna(0)

print("\nUser-item matrix shape:")
print(user_item_matrix.shape)

# Calculate cosine similarity between users.
collaborative_similarity_matrix = cosine_similarity(user_item_matrix)

collaborative_similarity = pd.DataFrame(
    collaborative_similarity_matrix,
    index=user_item_matrix.index,
    columns=user_item_matrix.index,
)

print("\nCollaborative similarity matrix shape:")
print(collaborative_similarity.shape)


# ---------------------------------------------------------
# Step 5 - Define recommendation helpers
# ---------------------------------------------------------

def normalize_scores(scores: pd.Series) -> pd.Series:
    """
    Scale scores to the range [0, 1] using min-max normalization.

    If all scores are equal, return zeros to avoid division by zero.
    """
    if scores.empty:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return pd.Series(0.0, index=scores.index)

    return (scores - minimum) / (maximum - minimum)


def recommend_movies(
    user_id: int,
    n_recommendations: int | None = 10,
    content_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Generate hybrid movie recommendations for a user.

    Args:
        user_id: ID of the target user.
        n_recommendations: Maximum number of movies to recommend.
        content_weight: Weight assigned to content-based scores.
            The collaborative weight is calculated as 1 - content_weight.

    Returns:
        A DataFrame with hybrid, content-based, and collaborative scores.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User not found: {user_id}")

    if n_recommendations is not None and n_recommendations < 1:
        raise ValueError("n_recommendations must be at least 1, or None for all candidates.")

    if not 0.0 <= content_weight <= 1.0:
        raise ValueError("content_weight must be between 0.0 and 1.0.")

    collaborative_weight = 1.0 - content_weight

    # Find movies that the user has already rated.
    user_ratings = user_item_matrix.loc[user_id]
    rated_movie_ids = user_ratings[user_ratings > 0].index

    # Identify movies the user has rated positively.
    liked_movie_ids = user_ratings[user_ratings >= 4.0].index

    # -----------------------------------------------------
    # Content-based scores
    # -----------------------------------------------------

    if len(liked_movie_ids) > 0:
        # Average each candidate movie's similarity to liked movies.
        content_scores = content_similarity.loc[
            :, liked_movie_ids
        ].mean(axis=1)
    else:
        # If the user has no highly rated movies, use neutral scores.
        content_scores = pd.Series(
            0.0,
            index=content_similarity.index,
        )

    # -----------------------------------------------------
    # Collaborative filtering scores
    # -----------------------------------------------------

    # Exclude the target user from the neighbor list.
    neighbor_similarities = collaborative_similarity.loc[user_id].drop(
        index=user_id
    )

    # Select the most similar users.
    neighbors = neighbor_similarities.nlargest(20)

    # Retrieve the ratings given by those neighbors.
    neighbor_ratings = user_item_matrix.loc[neighbors.index]

    # Use only positive similarity weights.
    positive_weights = neighbors.clip(lower=0)

    weighted_ratings = neighbor_ratings.mul(
        positive_weights,
        axis=0,
    )

    # Do not treat missing ratings as dislikes.
    rated_mask = neighbor_ratings.gt(0)
    weighted_rating_sum = weighted_ratings.sum(axis=0)
    rating_weight_sum = rated_mask.mul(
        positive_weights,
        axis=0,
    ).sum(axis=0)

    collaborative_scores = (
        weighted_rating_sum
        .div(rating_weight_sum.replace(0, float("nan")))
        .fillna(0.0)
    )

    # -----------------------------------------------------
    # Combine the scores
    # -----------------------------------------------------

    candidate_movie_ids = genre_features.index.difference(
        rated_movie_ids
    )

    content_scores = content_scores.reindex(
        candidate_movie_ids,
        fill_value=0.0,
    )
    collaborative_scores = collaborative_scores.reindex(
        candidate_movie_ids,
        fill_value=0.0,
    )

    normalized_content = normalize_scores(content_scores)
    normalized_collaborative = normalize_scores(collaborative_scores)

    hybrid_scores = (
        content_weight * normalized_content
        + collaborative_weight * normalized_collaborative
    )

    recommendations = pd.DataFrame(
        {
            "content_score": normalized_content,
            "collaborative_score": normalized_collaborative,
            "hybrid_score": hybrid_scores,
        }
    )

    recommendations = recommendations.sort_values(
        "hybrid_score",
        ascending=False,
    )

    if n_recommendations is not None:
        recommendations = recommendations.head(n_recommendations)

    # Add movie titles and genres to the result.
    recommendations = recommendations.join(
        movies.set_index("movieId")[["title", "genres"]]
    )

    recommendations.index.name = "movieId"

    return recommendations.reset_index()[
        [
            "movieId",
            "title",
            "genres",
            "content_score",
            "collaborative_score",
            "hybrid_score",
        ]
    ]


# ---------------------------------------------------------
# Step 6 - Generate and compare recommendations
# ---------------------------------------------------------

sample_user_id = 1

# Retrieve scores for all candidate movies so each method can be ranked
# independently.
all_recommendations = recommend_movies(
    user_id=sample_user_id,
    n_recommendations=None,
    content_weight=0.4,
)

print(f"\\nHybrid recommendations for user {sample_user_id}")
print(
    all_recommendations.sort_values("hybrid_score", ascending=False)
    .head(10)
    .to_string(index=False)
)

print(f"\\nContent-based recommendations for user {sample_user_id}")
print(
    all_recommendations.sort_values("content_score", ascending=False)
    [["title", "genres", "content_score"]]
    .head(10)
    .to_string(index=False)
)

print(f"\\nCollaborative recommendations for user {sample_user_id}")
print(
    all_recommendations.sort_values("collaborative_score", ascending=False)
    [["title", "genres", "collaborative_score"]]
    .head(10)
    .to_string(index=False)
)


# Step 7 - Experiment with different hybrid weights
# ---------------------------------------------------------

print("\nComparing different content weights")

for weight in [0.2, 0.4, 0.6, 0.8]:
    results = recommend_movies(
        user_id=sample_user_id,
        n_recommendations=5,
        content_weight=weight,
    )

    print(f"\nContent weight: {weight:.1f}")
    print(
        results[["title", "hybrid_score"]].to_string(index=False)
    )
