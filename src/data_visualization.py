import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# 保存图片的路径
OUTPUT_DIR = '../output/visualizations'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_plot(fig, filename):
    """保存图表到指定路径"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)  # 关闭图表，避免显示

def plot_time_series(df, columns):
    """绘制时间序列图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in columns:
        ax.plot(df['datetime'], df[col], label=col)
    ax.set_title('Air Quality Time Series')
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend(loc='upper left')
    save_plot(fig, 'time_series.png')

def plot_corr_matrix(df):
    """绘制相关性矩阵"""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_title('Correlation Matrix', pad=20)
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticklabels(df.columns)
    save_plot(fig, 'correlation_matrix.png')

def plot_interactive_time_series(df, columns):
    """绘制交互式时间序列图（使用 Plotly）"""
    # 创建交互式时间序列图
    fig = px.line(df, x='datetime', y=columns, title="Interactive Time Series", labels={'datetime': 'Time'})
    fig.update_traces(mode='lines')
    
    # 显示图表（会自动在浏览器中打开）
    fig.show()

def plot_air_quality_distribution(df, column):
    """绘制空气质量分布"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df[column], bins=50, color='blue', alpha=0.7)
    ax.set_title(f'{column} Distribution')
    ax.set_xlabel(f'{column} Concentration')
    ax.set_ylabel('Frequency')
    save_plot(fig, f'{column}_distribution.png')

def plot_boxplot(df, column):
    """绘制PM2.5箱线图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(df[column], vert=False)
    ax.set_title(f'{column} Boxplot')
    ax.set_xlabel(f'{column} Concentration')
    save_plot(fig, f'{column}_boxplot.png')

def visualize_data(file_path):
    """加载并可视化数据"""
    # 加载数据
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])  # 转换为时间格式

    # 绘制时间序列图
    plot_time_series(df, ['PM2.5', 'PM10', 'NO2', 'CO'])
    
    # 绘制相关性矩阵
    plot_corr_matrix(df)
    
    # 绘制交互式时间序列图（直接显示在浏览器中）
    plot_interactive_time_series(df, ['PM2.5', 'PM10', 'CO'])
    
    # 绘制空气质量分布图
    plot_air_quality_distribution(df, 'PM2.5')
    
    # 绘制PM2.5箱线图
    plot_boxplot(df, 'PM2.5')

if __name__ == "__main__":
    file_path = '../data/processed_data.csv'
    visualize_data(file_path)