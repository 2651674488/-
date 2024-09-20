import pandas as pd
import numpy as np




def build_movie_infos_ch(ratings_data_path, movies_data_path, save_path):
    # 载入数据
    ratings_data = pd.read_csv(ratings_data_path)
    movies_data = pd.read_csv(movies_data_path)
    movies_data = movies_data[['movieId', 'name', 'year', 'genres']]
    # 删除有空值的行
    movies_data.dropna(how='any', axis=0, inplace=True)
    # 计算电影的评分次数和电影的平均得分
    # 转为整数
    ratings_data['rating'] = ratings_data['rating'].astype(int)
    ratings_data = ratings_data[['userId', 'movieId', 'rating']]
    # 统计电影id出现次数
    ratings_count_df = ratings_data.value_counts(subset=['movieId']).sort_index().to_frame().reset_index()
    ratings_count_df.columns = ['movieId', 'rating_count']
    # 根据id分组计算平均值
    ratings_mean_df = ratings_data.groupby(by='movieId')['rating'].mean()
    ratings_mean_df = ratings_mean_df.to_frame().reset_index()
    count_rating_df = pd.merge(left=ratings_count_df, right=ratings_mean_df, on=['movieId'])
    items_data = pd.DataFrame()
    items_data['movieId'] = movies_data['movieId']
    items_data['name'] = movies_data['name']
    items_data['year'] = movies_data['year']
    items_data = items_data.dropna()
    items_data['year'] = items_data['year'].astype(int)
    items_data.reset_index(drop=True, inplace=True)


    # 流程类型，提取所有流派
    genres_set = set()
    for row in movies_data.itertuples():
        genres = row.genres.split("/")
        # 并集操作
        genres_set = set(genres_set) | set(genres) 
    genres_metrix = np.zeros((len(items_data), len(genres_set)), dtype=int)
    genres_df = pd.DataFrame(genres_metrix, columns=list(genres_set))
    # 电影信息与题材按照列合并
    genres_df = pd.concat([items_data, genres_df], axis=1)
    # 电影分数与题材合并
    items_df = pd.merge(left=count_rating_df, right=genres_df, on=['movieId'])
    
    # 设置每部电影的类型
    for row in movies_data.itertuples():
        genres = row.genres.split("/")
        items_df.loc[items_df['movieId'] == row.movieId, genres] = 1

    items_df.to_csv(save_path, index=None)
    return items_df


def load_movie_infos(movie_info_path, ratings_data_path, movies_data_path):
    try:
        return pd.read_csv(movie_info_path)
    except:
        print("电影信息未找到，正在生成电影信息，请稍后...")
        # 生成电影信息
        movie_infos = build_movie_infos_ch(ratings_data_path, movies_data_path, movie_info_path)
        print("电影信息建立完成")
        return movie_infos




def load_data(data_path):
    # 设置字段类型和所需列
    # dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    # cols = ['userId', 'movieId', 'rating']
    # 加载数据集
    # ratings_df = pd.read_csv(data_path, dtype=dtype)
    ratings_df = pd.read_csv(data_path)
    ratings_df = ratings_df[['userId', 'movieId', 'rating']]
    
    ratings_df['userId'] = ratings_df['userId'].astype(np.int32)
    ratings_df['movieId'] = ratings_df['movieId'].astype(np.int32)
    ratings_df['rating'] = ratings_df['rating'].astype(np.float32)

    return ratings_df

def load_user_list(data_path):
    df = load_data(data_path)
    # 基于userId删除重复行,通过索引选取userId列，并将其转换为列表
    user_list = df.drop_duplicates(subset=['userId'])['userId'].to_list()
    user_list = [str(i) for i in user_list]
    return user_list


def build_train_data(train_df):
    from surprise import Reader
    from surprise import Dataset

    ratings_reader = Reader(rating_scale=(1, 5))
    # 根据提供的reader对象将数据转换为适用于推荐系统的格式
    train_data = Dataset.load_from_df(train_df, reader=ratings_reader)

    return train_data



def build_model(train_data):
    from surprise import SVD
    from surprise.model_selection import GridSearchCV
    import time
    # 迭代次数（n_epochs）、学习率（lr_all）和正则化参数（reg_all）
    param_grid = {'n_epochs': [30, 45, 50], 'lr_all': [0.02, 0.05], 'reg_all': [0.15, 0.2]}
    # 创建了一个GridSearchCV对象gs，传入SVD模型、参数网格字典、评价指标（均方根误差RMSE和平均绝对误差MAE）以及交叉验证折数（cv=3）。这样就可以通过网格搜索在给定参数范围内找到最佳的模型参数
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    # 寻找最佳训练模型
    gs.fit(train_data)
    # 获取通过网格搜索得到的最佳模型，通过索引'rmse'获取对应的SVD模型
    start_time = time.time()
    svd_model = gs.best_estimator['rmse']
    # 使用build_full_trainset方法从训练数据集中构建一个适用于SVD模型的完整训练集
    fit_data = train_data.build_full_trainset()
    # 使用构建的完整训练集对SVD模型进行训练，即拟合模型
    svd_model.fit(trainset=fit_data)
    end_time = time.time()
    print(f"SVD模型运行时间: {end_time - start_time}秒")
    print(f"SVD的最佳超参数: {gs.best_params}")
    return svd_model

def build_SVDpp_model(train_data):
    from surprise import SVDpp
    from surprise.model_selection import GridSearchCV
    import time
    # 迭代次数（n_epochs）、学习率（lr_all）和正则化参数（reg_all）
    param_grid = {'n_epochs': [30, 45, 50], 'lr_all': [0.02, 0.05], 'reg_all': [0.15, 0.2]}
    # 创建了一个GridSearchCV对象gs，传入SVD模型、参数网格字典、评价指标（均方根误差RMSE和平均绝对误差MAE）以及交叉验证折数（cv=3）。这样就可以通过网格搜索在给定参数范围内找到最佳的模型参数
    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
    # 寻找最佳训练模型
    gs.fit(train_data)
    # 获取通过网格搜索得到的最佳模型，通过索引'rmse'获取对应的SVD模型
    svdpp_model = gs.best_estimator['rmse']
    # 使用build_full_trainset方法从训练数据集中构建一个适用于SVD模型的完整训练集
    fit_data = train_data.build_full_trainset()
    # 使用构建的完整训练集对SVD模型进行训练，即拟合模型
    svdpp_model.fit(trainset=fit_data)
    return svdpp_model
def build_KNNBasic_model(train_data):
    from surprise import accuracy
    from surprise.model_selection import KFold
    from surprise import KNNBasic
    import time
    # 相似度属性：pearson_baseline, cosine, euclidean(msd)
    # 基于物品的协同过滤
    # 通过设置 min_support，排除评分用户数量很少的物品
    algo = KNNBasic(k=50, sim_options={'user_based': False, 'name': 'msd', 'min_support': 5})
    # 定义K折交叉验证迭代器，K=3
    n_splits = 3
    avg_rmse = 0
    start_time = time.time()
    kf = KFold(n_splits=n_splits)
    for trainset, testset in kf.split(train_data):
        # 训练并预测
        algo.fit(trainset)
        predictions = algo.test(testset)
        # 计算RMSE
        avg_rmse += accuracy.rmse(predictions, verbose=True)
    end_time = time.time()
    print(f'基于物品的协同过滤的时间:{end_time - start_time}')
    avg_rmse = avg_rmse / n_splits
    print(f'基于物品的协同过滤的rmse平均值:{avg_rmse}')
    # 相似度属性：pearson_baseline, cosine, euclidean(msd)
    # 基于用户的协同过滤
    algo = KNNBasic(k=50, sim_options={'user_based': True, 'name': 'msd', 'min_support': 5})
    # 定义K折交叉验证迭代器，K=3
    n_splits = 3
    avg_rmse = 0
    start_time = time.time()
    kf = KFold(n_splits=n_splits)
    for trainset, testset in kf.split(train_data):
        # 训练并预测
        algo.fit(trainset)
        predictions = algo.test(testset)
        # 计算RMSE
        avg_rmse += accuracy.rmse(predictions, verbose=True)
    end_time = time.time()
    print(f'基于用户的协同过滤的时间:{end_time - start_time}')
    avg_rmse = avg_rmse / n_splits
    print(f'基于用户的协同过滤的rmse平均值:{avg_rmse}')


def save_model(model, save_path):
    from surprise.dump import dump
    try:
        # 使用dump函数将模型对象model保存到指定的文件路径save_path中。algo=model表示要保存的算法对象是model
        dump(save_path, algo=model)
    except Exception as e:
        print("保存失败，请再试一次吧")


def load_model(model_path):
    from surprise.dump import load
    try:
        model = load(model_path)
    except:
        return None
    else:
        return model


# train_data表示训练数据集，model表示已训练好的推荐系统模型
# 对所有用户进行评分预测
def predict_all(train_data, model):
    # 从训练数据集中构建一个适用于模型的完整训练集
    fit_data = train_data.build_full_trainset()
    # 构建一个测试集，其中包含了所有用户对所有未评分项的评分预测
    predicting_set = fit_data.build_anti_testset()
    # 使用已训练好的模型model对测试集predicting_set进行评分预测，得到一个评分预测列表
    prediction_list = model.test(predicting_set)

    return prediction_list

# 构建用户预测评分的字典，并按照评分进行排序
def build_user_prediction(prediction_list):
    # 创建一个默认值为列表的字典
    from collections import defaultdict
    # 这行创建了一个名为prediction_dict的defaultdict对象，用于保存预测评分的字典。字典的键是用户ID，值是一个列表，用于存储该用户的电影预测评分。
    prediction_dict = defaultdict(list)
    for p in prediction_list:
        prediction_dict[str(p.uid)].append((p.iid, p.est))
    sorted_prediction_dict={}
    # 遍历其中的每个用户和对应的电影评分列表
    for user_id, item_ratings in prediction_dict.items():
        # 对电影评分列表进行排序，按照评分（x[1]）进行降序排序
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        # 将排序后的电影评分列表保存到sorted_prediction_dict字典中，以用户ID作为键。
        sorted_prediction_dict[user_id] = item_ratings
    return sorted_prediction_dict


def save_predictions(prediction_dict, predictions_path):
    # Output as json file
    import json
    with open(predictions_path, 'w') as f:
        # 将prediction_dict字典的内容以JSON格式写入到文件对象f中
        json.dump(prediction_dict, f)

    return prediction_dict


def load_predictions(predictions_path):
    import json
    try:
        with open(predictions_path) as f:
            predicted_ratings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError
    else:
        return predicted_ratings



if __name__ == '__main__':
    RATINGS_DATA_PATH = "./ml-latest-small/ch_ratings_small.csv"
    MOVIES_DATA_PATH = "./ml-latest-small/ch_movies_small.csv"

    MODEL_PATH = "saved_files/model.plf"
    PREDICTIONS_PATH = "saved_files/ch_prediction_small.json"
    MOVIE_INFO_PATH = "saved_files/ch_movie_info_small.csv"
    build_movie_infos_ch(RATINGS_DATA_PATH, MOVIES_DATA_PATH, MOVIE_INFO_PATH)
    # print(load_user_list(RATINGS_DATA_PATH))
