import recomender_core as rc
from tkinter import messagebox
def make_data(data_path):
    print("加载数据中...")
    ratings_df = rc.load_data(data_path)
    train_data = rc.build_train_data(ratings_df)
    print("数据已经准备好")
    return train_data

def make_model(train_data, model_path):
    print("正在建模...这需要几分钟，请耐心等候...")
    model = rc.build_model(train_data)
    rc.save_model(model, model_path)
    print("建模完成")
    return model

def make_SVDpp_model(train_data, model_path):
    model = rc.build_SVDpp_model(train_data)
    rc.save_model(model, model_path)
    return model
def make_SlopeOne_model(train_data, model_path):
    model = rc.build_SlopeOne_model(train_data)
    rc.save_model(model, model_path)
    return model
def make_KNNBasic_model(train_data, model_path):
    model = rc.build_KNNBasic_model(train_data)
    rc.save_model(model, model_path)
    return model
def make_NMF_model(train_data, model_path):
    model = rc.build_NMF_model(train_data)
    rc.save_model(model, model_path)
    return model

def make_predictions(train_data, model, predictions_path):
    print("正在构建评分字典...这需要几分钟，请耐心等候...")
    # 对所有用户进行评分预测
    predict_list = rc.predict_all(train_data, model)
    # 构建用户预测评分的字典，并按照评分进行排序
    predictions = rc.build_user_prediction(predict_list)
    # 以json保存
    rc.save_predictions(predictions, predictions_path)
    print("构建完成")
    return predictions


def top_k_recommand(user_id, k, predicted_ratings, times):
    # user_id表示用户ID，k表示要推荐的电影数量，predicted_ratings表示评分预测列表，times表示推荐次数
    position = times * k + 1
    try:
        if predicted_ratings:
            # 从评分预测列表中根据用户ID，取出连续的k个电影作为推荐结果
            movie_list = predicted_ratings[user_id][position:position+k]
        else:
            # 当评分预测列表为空时，从某个地方加载评分预测结果。这里使用了rc.load_predictions()方法来加载评分预测结果
            predicted_ratings = rc.load_predictions()
            # 从加载的评分预测结果中根据用户ID和起始位置，取出连续的k个电影作为推荐结果
            movie_list = predicted_ratings[user_id][position:position+k]
        return movie_list
    except KeyError:
        messagebox.showwarning(title='提示', message='新用户注册后需要更新推荐，请点击更新推荐')


# if __name__ == '__main__':
#     DATA_PATH = "ml-latest-small/ratings.csv"
#     MODEL_PATH = "saved_files/model.plf"
#     PREDICTIONS_PATH = "saved_files/prediction.json"
#
#     try:
#         print("加载预测文件...")
#         predictions = rc.load_predictions(PREDICTIONS_PATH)
#     except FileNotFoundError:
#         print("预测文件未找到，正在建模，请耐心等待...")
#         train_data = make_data(DATA_PATH)
#         model = make_model(train_data, MODEL_PATH)
#         predictions = make_predictions(train_data, model, PREDICTIONS_PATH)
