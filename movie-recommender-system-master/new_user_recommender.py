import pandas as pd


def select_movies_by_genres(movie_info_df, user_genres_list, k):
    filtered_df = movie_info_df.loc[
        movie_info_df[user_genres_list].isin([1]).any(axis=1)
    ][["movieId", "name", "year", "rating", "rating_count"]]

    results_df = filtered_df.sort_values(
        by=["rating", "rating_count"], ascending=False
    ).reset_index()
    # results = results_df.loc[results_df['rating'] > 4].sample(n=k, replace=True)
    if len(results_df) < k:
        k = len(results_df)
    results = results_df.sample(n=k, replace=True)
    results_list = results["movieId"].to_list()

    return results_list




def add_rating(userId, rating, movieId):
    RATINGS_DATA_PATH = "./ml-latest-small/ch_ratings_small.csv"
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    # pd.Timestamp.now()
    ratings_df.append(movieId, rating, userId)
    ratings_df.to_csv(RATINGS_DATA_PATH, index=None)
    print("添加成功")


if __name__ == "__main__":
    RATINGS_DATA_PATH = "ml-latest-small/ratings.csv"
    add_rating(RATINGS_DATA_PATH, 1, "Heat", 5)
