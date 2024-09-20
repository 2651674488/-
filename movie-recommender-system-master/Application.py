import pandas as pd

import recomender_core as rc
import svd_recomander as sr
import new_user_recommender as nur
import time

RATINGS_DATA_PATH = "./ml-latest-small/ch_ratings_small.csv"
MOVIES_DATA_PATH = "./ml-latest-small/ch_movies_small.csv"

MODEL_PATH = "saved_files/model.plf"
PREDICTIONS_PATH = "saved_files/ch_prediction_small.json"
MOVIE_INFO_PATH = "saved_files/ch_movie_info_small.csv"

def compare_model():
    train_data = sr.make_data(RATINGS_DATA_PATH)

    # SVD
    # sr.make_model(train_data, MODEL_PATH)

    # KNNBasic基本协同过滤
    sr.make_KNNBasic_model(train_data, MODEL_PATH)

    # sr.make_predictions(train_data, model, PREDICTIONS_PATH)

def login_initialisation():
    train_data = sr.make_data(RATINGS_DATA_PATH)
    model = sr.make_model(train_data, MODEL_PATH)
    # prediction是预测评分字典
    sr.make_predictions(train_data, model, PREDICTIONS_PATH)

# 初始化，训练模型
def initialisation():
    try:
        print("加载预测文件...")
        predictions = rc.load_predictions(PREDICTIONS_PATH)
    except FileNotFoundError:
        # 没有文件，重新训练
        print("未找到预测文件，正在进行预测，请稍后...")
        train_data = sr.make_data(RATINGS_DATA_PATH)
        model = sr.make_model(train_data, MODEL_PATH)
        # prediction是预测评分字典
        predictions = sr.make_predictions(train_data, model, PREDICTIONS_PATH)
    return predictions


# 返回电影id和对应name
def convert_movie_name(movie_id_list):
    movies_df = pd.read_csv(MOVIES_DATA_PATH)
    # 使用DataFrame的isin方法来筛选出列表中的movieId对应的所有行
    filtered_movies_df = movies_df[movies_df['movieId'].isin(movie_id_list)]
    # 这里选择将movieId作为索引，转为字典
    movies_dict = filtered_movies_df.set_index('movieId').T.to_dict()

    # 将字典转换为JSON格式的字符串
    import json
    movies_json = json.dumps(movies_dict, indent=4, ensure_ascii=False)

    # 输出JSON格式的字符串
    return movies_json

def add_rating(uid, rating, movieId):
    df = pd.read_csv(RATINGS_DATA_PATH)
    # 检查特定的user_id和movie_id组合是否存在
    if df[(df['userId'] == uid) & (df['movieId'] == movieId)].empty:
        # 如果不存在，添加新行
        new_row = {'userId': uid, 'movieId': movieId, 'rating': rating}
        df = df.append(new_row, ignore_index=True)
    else:
        # 如果存在，更新评分
        df.loc[(df['userId'] == uid) & (df['movieId'] == movieId), 'rating'] = rating
    # 将更新后的DataFrame写回CSV文件
    df.to_csv(RATINGS_DATA_PATH, index=False)

def prediction(userId, num_k, times):
    # 建立模型，给出json
    predicted_ratings = initialisation()
    recomending_list = sr.top_k_recommand(userId, num_k, predicted_ratings, times)
    result_list = [x[0] for x in recomending_list]
    result_json = convert_movie_name(result_list)
    # 返回推荐电影的json
    return result_json



# 新用户推荐
def new_user_recommend(prefered_genres):
    movie_info_df = rc.load_movie_infos(MOVIE_INFO_PATH, RATINGS_DATA_PATH, MOVIES_DATA_PATH)
    recommend_list = nur.select_movies_by_genres(movie_info_df, prefered_genres, 1)
    result_json = convert_movie_name(recommend_list)
    return result_json

if __name__ == '__main__':
    compare_model()