# 项目信息
__project__ = "城市空气质量预测项目"
__version__ = "1.3"
__author__ = "wyw"
__description__ = """
本项目通过机器学习方法，基于历史空气质量数据，预测未来几个小时的空气污染物浓度。
主要使用 RNN、LSTM 和 GRU 等循环神经网络模型对数据进行建模和预测。
"""

# 打印项目信息
def print_project_info():
    print(f"----- {__project__} -----")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")

# 调用打印函数，初始化时自动输出项目信息
print_project_info()