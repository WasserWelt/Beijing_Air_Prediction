import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(df, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    将数据集划分为训练集、验证集和测试集，保持时间顺序。
    """
    # 根据时间顺序划分数据
    train_end = int(len(df) * train_size)
    valid_end = int(len(df) * (train_size + valid_size))

    train_data = df[:train_end]
    valid_data = df[train_end:valid_end]
    test_data = df[valid_end:]

    return train_data, valid_data, test_data

import numpy as np

def prepare_data(df, look_back, forecast_horizon, target_columns=["PM2.5"]):
    """
    将数据处理为适合时间序列预测的格式
    :param df: 原始的DataFrame
    :param look_back: 用来预测的过去时间步数（例如24小时）
    :param forecast_horizon: 预测的未来时间步数（例如1小时）
    :param target_columns: 需要预测的污染物列名（如 ["PM2.5", "CO"]）
    :return: 训练集、验证集和测试集的输入输出数据
    """
    # 去掉时间列，只保留特征
    data = df.drop('datetime', axis=1).values  # 确保只有数值数据
    data = data.astype(np.float32)  # 显式转换为 float32 类型

    # 获取目标列的索引
    target_indices = [df.columns.get_loc(col) for col in target_columns]
    
    X, y = [], []

    # 创建滑动窗口
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i+look_back])  # past `look_back` hours of data
        y.append(data[i+look_back+forecast_horizon-1, target_indices])  # 多列污染物预测

    X = np.array(X, dtype=np.float32)  # 显式转换为 float32 类型
    y = np.array(y, dtype=np.float32)  # 显式转换为 float32 类型

    # 划分数据集
    train_size = int(len(X) * 0.8)
    valid_size = int(len(X) * 0.1)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_valid, y_valid = X[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
    X_test, y_test = X[train_size+valid_size:], y[train_size+valid_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
