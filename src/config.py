import argparse
import os
import yaml
'''
def get_config():
    """
    获取命令行参数和配置文件中的超参数
    """
    parser = argparse.ArgumentParser(description="Air Quality Prediction")

    # 模型超参数
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm', 'gru'], default='lstm', help="选择模型类型")
    parser.add_argument('--epochs', type=int, default=20, help="训练轮次")
    parser.add_argument('--batch_size', type=int, default=32, help="批量大小")
    parser.add_argument('--lr', type=float, default=0.001, help="学习率")
    
    # 配置文件路径
    parser.add_argument('--config', type=str, default='config.yaml', help="配置文件路径")

    # 加载参数
    args = parser.parse_args()

    # 如果有配置文件，加载配置文件中的参数
    if args.config:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')  # 根目录路径
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # 加载配置文件中的参数
        args.model = config.get('model', args.model)
        args.epochs = config.get('epochs', args.epochs)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.lr = config.get('lr', args.lr)
        args.target_columns = config.get('target_columns', [])  # 添加target_columns
        
    return args
'''

def get_config():
    """
    获取命令行参数和配置文件中的超参数
    """
    # 配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')  # 根目录路径

    # 加载配置文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config
