import logging
import csv
import os

def setup_logging(log_file='output/train_log.csv'):
    """
    设置日志记录器，保存训练过程中的结果到 CSV 文件
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 写入标题
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['model', 'epochs', 'batch_size', 'lr', 'train_loss', 'valid_loss', 'final_mse'])

def save_training_result(model_tag, train_loss=None, valid_loss=None, mse=None, file_path='./output/train_log.csv'):
    """保存训练结果"""
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if model_tag != 'final_mse':
            writer.writerow([model_tag, '', '', '', train_loss, valid_loss, ''])
        else:
            # 保存 final_mse
            writer.writerow([model_tag, '', '', '', '', '', mse])

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)
