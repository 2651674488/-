import numpy as np
import pandas as pd
import math

from sklearn import model_selection
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise import SVD
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

import recomender_core as rc

# top_n_dict，表示用户的电影推荐字典
# 根据用户的电影推荐字典，计算用户电影推荐的多样性得分，并返回平均多样性得分
def calculate_diversity(top_n_dict):
    diversity = 0
    # 返回字典top_n_dict中的所有键，即用户ID
    for userId in top_n_dict.keys():
        # 获取用户ID对应的电影推荐列表，并提取出电影ID构成的列表top_n_list
        top_n_list = [x[0] for x in top_n_dict[userId]]
        # 根据电影ID列表过滤电影信息表movie_info，得到用户推荐的电影信息表top_n_movies。其中，isin(top_n_list)用于判断电影ID是否存在于top_n_list中，drop(['movieId', 'name'], axis=1)用于删除表中的'movieId'和'name'列？
        top_n_movies = movie_info.loc[movie_info['movieId'].isin(top_n_list)].drop(['movieId', 'name'], axis=1)
        # 对电影信息进行标准化
        scaler = StandardScaler()
        # 使用标准化器scaler对电影信息表top_n_movies进行拟合，计算出均值和标准差
        scaler.fit(top_n_movies)
        # 使用标准化器scaler对电影信息表top_n_movies进行标准化转换，得到标准化后的结果result
        result = scaler.transform(top_n_movies)
        # 计算标准化结果result的欧氏距离，并将平均距离累加到多样性得分diversity
        diversity += euclidean_distances(result).mean()
    # 返回多样性得分的平均值，即用户电影推荐的多样性
    return diversity / len(top_n_dict)


def evaluate_personalised(train_df, test_df, n):
    ratings_reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df, reader=ratings_reader)

    param_grid = {'n_epochs': [45, 50, 55], 'lr_all': [0.02, 0.05], 'reg_all': [0.15, 0.2]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    # 在训练数据集上进行网格搜索，找到最佳超参数组合
    gs.fit(train_data)
    # 获取具有最佳RMSE评分的SVD模型
    svd_model = gs.best_estimator['rmse']
    fit_data = train_data.build_full_trainset()
    # 使用最佳超参数组合训练SVD模型
    svd_model.fit(trainset=fit_data)

    # 定义一个名为test_predict的嵌套函数，用于计算个性化评分
    def test_predict(df, model):
        # 使用模型对测试数据集进行预测，得到个性化评分
        estimated_rating = model.predict(df['userId'], df['movieId']).est
        # print(estimated_rating)
        return estimated_rating
    # 将个性化评分应用到测试数据集的每一行，并将结果保存在新的列personalised_rating中
    test_df['personalised_rating'] = test_df.apply(test_predict, model=svd_model, axis=1)
    # 计算个性化评分的均方根误差（RMSE）
    rmse = math.sqrt(mean_squared_error(test_df['rating'], test_df['personalised_rating']))
    print("Personalised RMSE: ", rmse)

    # 使用训练数据集构建的反测试集对模型进行预测，得到推荐结果
    predictions = svd_model.test(testset=fit_data.build_anti_testset())
    # 创建一个默认列表字典prediction_dict，用于保存预测的评分结果
    prediction_dict = defaultdict(list)
    for p in predictions:
        # 将每个用户ID对应的预测评分结果添加到prediction_dict
        prediction_dict[p.uid].append((p.iid, p.est))
    # 创建一个空字典top_n_dict，用于保存每个用户的top n推荐结果
    top_n_dict = {}
    for user_id, item_ratings in prediction_dict.items():
        # 对评分结果进行排序，按照评分从高到低排序
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        # 将排序后的前n个评分结果存储在top_n_dict中，作为每个用户的top n推荐
        top_n_dict[user_id] = item_ratings[:n]

    diversity = calculate_diversity(top_n_dict)
    print("Personalised diversity: ", diversity)

# 非个性化推荐
def evaluate_non_personalised(train_df, test_df, n, by="rating"):
    # 如果by为"rating"，则将训练数据集按电影分组，并计算每部电影的平均评分
    if by == "rating":
        non_personalised_movie_rating_dict = pd.cut(ratings_df.groupby(by=['movieId'])['rating'].mean(), bins=10,
                                                labels=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]).to_dict()
    else:
        non_personalised_movie_rating_dict = pd.cut(ratings_df['movieId'].value_counts(), bins=10,
                                                labels=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]).to_dict()

    def set_score(df, dict):
        predicted_rating = non_personalised_movie_rating_dict[df['movieId']]
        return predicted_rating
    test_df['non_personalised_rating'] = test_df.apply(set_score, dict=non_personalised_movie_rating_dict, axis=1)
    rmse = math.sqrt(mean_squared_error(test_df['rating'], test_df['non_personalised_rating']))
    print("Non-personalised RMSE: ", rmse)

    non_personalised_movie_rating_list = sorted(non_personalised_movie_rating_dict.items(), key=lambda d: d[1],
                                                reverse=True)
    test_user_list = test_df.drop_duplicates(subset=['userId'])['userId'].to_list()
    non_personalised_top_n_dict = {}
    for user in test_user_list:
        non_personalised_top_n_dict[user] = non_personalised_movie_rating_list[:n]

    diversity = calculate_diversity(non_personalised_top_n_dict)
    print("Non-personalised diversity: ", diversity)



if __name__ == '__main__':
    DATA_PATH = "./ml-latest-small/ch_ratings_small.csv"
    MOVIES_DATA_PATH = "./ml-latest-small/ch_movies_small.csv"
    MOVIE_INFO_PATH = "saved_files/ch_movie_info_small.csv"

    dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    cols = ['userId', 'movieId', 'rating']

    ratings_df = pd.read_csv(DATA_PATH, dtype=dtype, usecols=cols)

    train_df, test_df = model_selection.train_test_split(ratings_df, test_size=0.2)
    movie_info = rc.load_movie_infos(MOVIE_INFO_PATH, DATA_PATH, MOVIES_DATA_PATH)

    n = 20
    evaluate_personalised(train_df, test_df, n)
    evaluate_non_personalised(train_df, test_df, n, "rating")
    evaluate_non_personalised(train_df, test_df, n, "popularity")