# 城市空气质量预测项目

## 项目概述

> 本项目来自华东理工大学机器学习课程设计

本项目致力于利用机器学习技术，基于历史空气质量数据，对未来几小时的空气污染物浓度进行预测，重点关注PM2.5、CO、NO2等污染物。我们采用循环神经网络（RNN）及其衍生模型（LSTM、GRU）对数据进行建模和预测。

项目涵盖数据预处理、特征选择、模型训练、评估与可视化等多个关键环节，并支持多种模型和超参数的配置调整，便于进行模型对比分析和优化。

## 技术栈

- **Python 3.x**
- **PyTorch**：深度学习框架
- **Pandas**：数据处理工具
- **NumPy**：科学计算库
- **Matplotlib/Seaborn**：数据可视化工具
- **yaml**：配置文件管理

## 项目目录结构

```bash
project/
│
├── data/                          # 存放原始和处理后的数据
│   ├── Beijingair.csv             # 原始数据文件
│   └── processed_data.csv         # 数据处理后文件
│
├── output/                        # 存放输出结果
│   ├── train_log.csv              # 训练日志文件
│   ├── visualizations/            # 可视化图片存放目录
│   └── results/                   # 可视化结果存放目录
├── src/                           # 项目源代码目录
│   ├── __init__.py                # 初始化
│   ├── data_preprocessing.py      # 数据预处理脚本
│   ├── data_visualization.py      # 数据可视化脚本
│   ├── model.py                   # 模型定义（RNN, LSTM, GRU）
│   ├── train.py                   # 训练和评估脚本
│   ├── train_log.py               # 日志记录和保存
│   ├── config.py                  # 配置文件加载
│   └── dataset_preparing.py       # 数据集划分与准备
│
├── config.yaml                    # 配置文件
└── main.py                        # 主程序入口
```

## 依赖安装

项目所需依赖可通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 配置文件说明

### 配置文件结构（config.yaml）

`config.yaml` 文件用于设置模型类型、训练超参数以及预测污染物。以下为配置示例：

```yaml
models:
  - name: rnn
    params:
      - epochs: 2
        batch_size: 32
        lr: 0.001
        target_columns:
          - 'PM2.5'
          - 'CO'
      - epochs: 3
        batch_size: 64
        lr: 0.0005
        target_columns:
          - 'PM2.5'
          - 'CO'
          - 'NO2'
  - name: lstm
    params:
      - epochs: 2
        batch_size: 32
        lr: 0.001
        target_columns:
          - 'PM2.5'
          - 'NO2'
  - name: gru
    params:
      - epochs: 2
        batch_size: 32
        lr: 0.001
        target_columns:
          - 'PM2.5'
          - 'CO'
```

**说明：**

- **models**：指定训练的模型类型（rnn、lstm、gru）。
- **params**：每种模型的超参数配置，包括训练轮次（epochs）、批量大小（batch_size）、学习率（lr）以及预测目标污染物（target_columns）。

**支持的预测项（target_columns）**

以下是支持的预测项列表，可根据需求在 `config.yaml` 中选择：

```yaml
target_columns:
    - 'PM2.5'
    - 'CO'
    - 'NO2'
    - 'SO2'
    - 'O3'
    - 'Temperature'
    - 'Pressure'
    - 'DewPoint'
    - 'WindSpeed'
```

## 使用指南

### 1. 数据预处理与可视化

数据预处理包括加载原始数据并转换为适合模型训练的格式。同时，可通过可视化功能对数据进行分析。具体实现和调用可在 `data_visualization.py` 中完成。执行：

```bash
python src/data_visualization.py
```

### 2. 模型训练

执行以下命令训练模型，模型和参数将根据 `config.yaml` 文件中的配置进行训练：

```bash
python main.py --config config.yaml
```

### 3. 模型评估

训练完成后，可以查看每个模型的评估结果（如均方误差 MSE）。评估结果将记录在日志文件中，便于后续对比分析。

### 4. 训练过程可视化

训练过程中，每个模型的损失值（包括训练集和验证集的损失）将被记录，并生成可视化图像。每个模型的训练损失图将保存在 `output/results` 目录下。通过查看不同模型在训练过程中损失的变化，可以进行对比分析和模型调优。

训练完成后，项目将自动生成每个模型对同一目标污染物（如 PM2.5、CO、NO2）的预测结果对比图。每张图展示了不同模型对测试集的预测结果与真实值的比较，有助于评估模型的预测性能。

每个模型的最终评估结果，包括均方误差（MSE），将被计算并展示。通过绘制不同模型的 MSE 值对比图，可以直观地看出哪个模型在预测任务中表现最佳。

## 日志与结果保存

训练过程中，所有训练和验证结果将保存至 `train_log.csv` 文件中。该文件包含每个模型的训练损失、验证损失以及最终的 MSE。

## 项目优化

- 你可以通过修改配置文件中的参数来调整模型，并进行不同的实验对比。
- 日志记录功能已实现，方便查看每次训练过程的详细情况。

## 许可证

MIT License
