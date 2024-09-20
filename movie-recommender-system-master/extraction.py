import pandas as pd

# 设置原数据路径
## 中文数据集
ch_movies_df = pd.read_csv("./ml-latest-small/ch_movies.csv")
ch_ratings_df = pd.read_csv("./ml-latest-small/ch_ratings.csv")
## 英文数据集
# en_movies = pd.read_csv('./ml-latest-small/en_movies.csv')
# en_ratings = pd.read_csv('./ml-latest-small/en_ratings.csv')

# genres_list包含了想要的电影类型, 如果电影类型中包含了genres_list中的任意一个, 则保留
genres_list = [
    "战争",
    "惊悚",
    "犯罪",
    "动画",
    "动作",
    "剧情",
    "爱情",
    "冒险",
    "悬疑",
    "科幻",
]

# 去掉genres为空的数据
ch_movies_df = ch_movies_df[ch_movies_df["genres"].notna()]
# 选取genres列中包含genres_list中的任意一个的数据
ch_movies_df["genres_list"] = ch_movies_df["genres"].apply(
    lambda x: any(i in x.split("/") for i in genres_list)
)

# 选取genres_list为True的数据
ch_movies_filtered_by_genres_df = ch_movies_df[ch_movies_df["genres_list"] == True]

# 选取前1000条电影数据
sub_movies_id_list = ch_movies_filtered_by_genres_df.loc[999:2999, "movieId"].to_list()
# 选取movieId在sub_movies_id_list中的数据
sub_raings_df = ch_ratings_df[ch_ratings_df["movieId"].isin(sub_movies_id_list)]

# 根据选取的数据, 选出movieId列的唯一值 (查看有什么电影)
filtered_movies_list = sub_raings_df["movieId"].drop_duplicates().to_list()

# 选取根据电影id, 在电影列中筛选出的电影数据
sub_movies_df = ch_movies_filtered_by_genres_df[
    ch_movies_filtered_by_genres_df["movieId"].isin(filtered_movies_list)
]


# 保存数据
sub_raings_df.to_csv("./ml-latest-small/ch_ratings_small.csv", index=False)
sub_movies_df.to_csv("./ml-latest-small/ch_movies_small.csv", index=False)
