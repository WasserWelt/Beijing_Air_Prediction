import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def create_dataloaders(X_train, y_train, X_valid, y_valid, batch_size=32):
    """
    将数据转化为 DataLoader
    """
    # 转换为 Tensor
    train_inputs_tensor = X_train.clone().detach().float()
    train_labels_tensor = y_train.clone().detach().float()
    valid_inputs_tensor = X_valid.clone().detach().float()
    valid_labels_tensor = y_valid.clone().detach().float()

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(train_inputs_tensor, train_labels_tensor)
    valid_dataset = TensorDataset(valid_inputs_tensor, valid_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


from src.train_log import setup_logging, save_training_result, log_info
import os

def train(model, X_train, y_train, X_valid, y_valid, epochs=20, lr=0.001, batch_size=32, save_dir='./output/results'):
    """
    训练模型并保存训练过程的图像
    """
    setup_logging('./output/train_log.csv')  # 初始化日志
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 获取训练集和验证集的 DataLoader
    train_loader, valid_loader = create_dataloaders(X_train, y_train, X_valid, y_valid, batch_size)

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # 训练集前向反向传播
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证集评估
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        # 输出每个 epoch 的损失
        log_info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

    # 可视化训练过程
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss', color='b')
    plt.plot(range(epochs), valid_losses, label='Validation Loss', color='r')
    plt.title(f'{model.__class__.__name__} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存训练损失图像
    model_name = f"{model.__class__.__name__}_epochs{epochs}_batch{batch_size}_lr{lr}"
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss.png"))
    plt.close()

    return model

def evaluate(model, X_test, y_test, batch_size=32):
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test.clone().detach().float(),
                                           y_test.clone().detach().float()), 
                             batch_size=batch_size, shuffle=False)
    
    mse_total = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            mse_batch = ((outputs - labels) ** 2).mean()  # 计算当前批次的 MSE
            mse_total += mse_batch.item()
            count += 1

    mse = mse_total / count  # 计算所有批次的平均 MSE
    return mse
