import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """读取数据并返回DataFrame"""
    df = pd.read_csv(file_path, header=None)
    #print("前几行数据：", df.head())  # 查看前几行数据
    df.columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temperature', 'Pressure', 'DewPoint', 'WindSpeed']
    return df

def clean_data(df):
    """处理缺失值及不需要的列"""
    # 假设数据里没有需要删除的列，如果有可以在这里进行删除
    # 例如：df = df.drop(columns=['某些列'], errors='ignore')
    
    # 检查并填充缺失值，可以选择删除或用均值填充
    df.fillna(df.mean(), inplace=True)  # 用均值填充
    return df

def normalize_data(df):
    """对数据进行标准化"""
    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temperature', 'Pressure', 'DewPoint', 'WindSpeed']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def add_time_column(df):
    """生成时间列并添加到DataFrame中"""
    start_date = '2013-03-01 00:00:00'  # 假设数据从2013年3月1日开始
    date_range = pd.date_range(start=start_date, periods=len(df), freq='H')  # 生成每小时的时间序列
    df['datetime'] = date_range
    return df

def preprocess_data(file_path, save_path='data/processed_data.csv'):
    """主函数：读取数据，清洗，标准化，生成时间列，并保存"""
    # 1. 加载数据
    df = load_data(file_path)
    # 2. 清洗数据
    df = clean_data(df)
    # 3. 数据标准化
    df, scaler = normalize_data(df)
    # 4. 添加时间列
    df = add_time_column(df)
    # 5. 保存处理后的数据
    df.to_csv(save_path, index=False)  # 保存为新的CSV文件，去掉索引列
    
    return df, scaler  # 返回处理后的数据和标准化器

# Example usage:
# df, scaler = preprocess_data('data/Beijingair.csv', save_path='data/processed_data.csv')
