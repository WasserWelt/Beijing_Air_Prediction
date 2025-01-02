import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.data_preprocessing import preprocess_data
from src.data_visualization import visualize_data
from src.model import RNNModel, LSTMModel, GRUModel
from src.train import train, evaluate
from src.train_log import setup_logging, log_info, save_training_result
from src.config import get_config
from src.dataset_preparing import split_data, prepare_data
import os


def main():
    # 设置日志
    setup_logging()

    # 加载配置参数
    config = get_config()

    # 数据预处理
    file_path = 'data/Beijingair.csv'
    save_path = 'data/processed_data.csv'
    log_info(f"数据预处理开始，加载文件: {file_path}")
    
    df, scaler = preprocess_data(file_path, save_path)
    log_info(f"数据预处理完成，数据已保存至 {save_path}")

    # 数据集划分
    train_data, valid_data, test_data = split_data(df)
    log_info(f"数据集划分完成：训练集{len(train_data)}，验证集{len(valid_data)}，测试集{len(test_data)}")

    # 使用滑动窗口处理数据
    look_back = 24  # 使用过去24小时的数据来预测未来污染物
    forecast_horizon = 1  # 预测未来1小时的PM2.5

    # 用于存储各模型的预测结果和MSE
    model_predictions = {}
    model_mse = {}

    # 遍历每个模型和其对应的参数
    for model_config in config['models']:
        model_name = model_config['name']
        
        # 遍历每个模型的不同超参数组合
        for params in model_config['params']:
            target_columns = params['target_columns']  # 获取目标污染物
            epochs = params['epochs']
            batch_size = params['batch_size']
            lr = params['lr']

            log_info(f"正在训练模型: {model_name} with epochs={epochs}, batch_size={batch_size}, lr={lr}, target_columns={target_columns}")
            
            # 数据准备
            X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(
                df, look_back, forecast_horizon, target_columns
            )
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

            # 选择模型
            if model_name == 'rnn':
                model = RNNModel(input_size=X_train.shape[2], hidden_size=50, output_size=len(target_columns))
            elif model_name == 'lstm':
                model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, output_size=len(target_columns))
            elif model_name == 'gru':
                model = GRUModel(input_size=X_train.shape[2], hidden_size=50, output_size=len(target_columns))

            # 训练模型
            trained_model = train(model, X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor,
                                  epochs=epochs, lr=lr, batch_size=batch_size)

            # 评估模型
            mse = evaluate(trained_model, X_test_tensor, y_test_tensor)
            model_mse[f"{model_name}_{epochs}epochs_{batch_size}bs_{lr}lr"] = mse  # 保存MSE
            save_training_result(f"{model_name}_{epochs}epochs_{batch_size}bs_{lr}lr_final_mse", mse=mse)  # 保存最终的 MSE
            log_info(f"{model_name} 模型训练完成，MSE: {mse:.4f}")

            # 保存模型的预测结果
            with torch.no_grad():
                model_predictions[model_name] = trained_model(X_test_tensor).numpy()
    # 画出每个模型在同一目标污染物上的预测结果对比
    for target_column in target_columns:
        plt.figure(figsize=(12, 8))
        true_values = y_test_tensor[:, target_columns.index(target_column)].numpy()

        # 调试输出，检查 predictions 的形状和目标污染物的索引
        print(f"Predictions Shape for {target_column}: {model_predictions[list(model_predictions.keys())[0]].shape}")
        print(f"Target Column Index for {target_column}: {target_columns.index(target_column)}")

        # 绘制真实数据
        plt.plot(true_values[:100], label='True Values', color='k', linewidth=2)

        for model_name, predictions in model_predictions.items():
            # 确保预测列数与目标污染物匹配
            if predictions.shape[1] > target_columns.index(target_column):
                model_target_preds = predictions[:, target_columns.index(target_column)]
                plt.plot(model_target_preds[:100], label=f'{model_name} Predictions')  # 只展示前100个预测点，避免混乱

        plt.title(f'{target_column} - Model Predictions vs True Values')
        plt.xlabel('Time Step')
        plt.ylabel(f'{target_column} Level')
        plt.legend()
        plt.grid(True)

        # 保存图像
        os.makedirs('./output/results', exist_ok=True)
        plt.savefig(f'./output/results/{target_column}_comparison_plot.png')
        plt.close()
    log_info(f"所有模型的预测结果对比图已保存至 './output/results' 目录")

    # 画出模型的MSE对比图
    mse_values = list(model_mse.values())
    model_names = list(model_mse.keys())

    # 绘制 MSE 对比图
    plt.figure(figsize=(10, 6))
    plt.barh(model_names, mse_values, color='skyblue')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.title('MSE Comparison Across Models')
    plt.grid(True, axis='x')

    # 保存图像
    os.makedirs('./output/results', exist_ok=True)
    plt.savefig('./output/results/mse_comparison_plot.png')
    plt.close()

    # 可视化完成提示
    log_info(f"所有模型的MSE对比图已保存至 './output/results' 目录")

if __name__ == '__main__':
    main()
