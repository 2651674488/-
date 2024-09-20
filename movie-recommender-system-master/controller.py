from tkinter import messagebox
import pandas as pd
import Application as ap

RATINGS_DATA_PATH = "ml-latest-small/ch_ratings_small.csv"
MOVIES_DATA_PATH = "ml-latest-small/ch_movies_small.csv"

def check_userid(uid):
    df = pd.read_csv(RATINGS_DATA_PATH)
    # 检查userId列中是否存在uid
    if df['userId'].isin([uid]).any():
        return True
    else:
        return False
def register():
    from datetime import datetime
    # 时间戳
    now = datetime.now()
    # 将datetime对象转换为时间戳（秒）
    timestamp = int(now.timestamp())
    # 将时间戳转换为字符串
    uid = str(timestamp)
    return uid
def rating_history(uid):
    # 返回打分历史json，根据uid给出{name, rating}
    ratings_df = pd.read_csv(RATINGS_DATA_PATH)
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    target_user_id = int(uid)

    user_ratings = ratings_df[ratings_df['userId'] == target_user_id]

    # 通过movieId将用户评分数据与电影信息数据关联
    user_movies = pd.merge(user_ratings, movies_df, on='movieId')

    # 选择需要的列，这里我们选择电影名称和评分
    user_movies_selected = user_movies[['name', 'rating']]

    # 将数据转换为字典格式，字典的键是电影名称，值是评分
    movies_dict = user_movies_selected.set_index('name')['rating'].to_dict()
    print(movies_dict)
    # 输出结果
    return movies_dict
def recommend_movie(uid, times):
    result_json = ap.prediction(uid, 1, times)
    # movieId	name	alias	actors	directors	doubanScore	genres	languages	mins	regions	storyline	tags	year
    import json
    data_dict = json.loads(result_json)
    # 获取第一个键（电影ID）和对应的值（电影信息字典）
    key, value = next(iter(data_dict.items()))
    # 将第一个键对应的值（电影信息）生成一个新的字典
    movie_info_dict = dict(value)
    first_key = list(data_dict.keys())[0]
    return first_key, movie_info_dict

def grade_movie(uid, movieId, rating):
    # 若存在，更新数据，不存在则添加数据
    ap.add_rating(uid, rating, movieId)

def new_user_recommend_movie(genres_list, uid, times):
    # 根据用户喜欢的类型，做推荐
    result_json = ap.new_user_recommend(genres_list)
    import json
    data_dict = json.loads(result_json)
    # 获取第一个键（电影ID）和对应的值（电影信息字典）
    key, value = next(iter(data_dict.items()))
    # 将第一个键对应的值（电影信息）生成一个新的字典
    movie_info_dict = dict(value)
    first_key = list(data_dict.keys())[0]
    return first_key, movie_info_dict

