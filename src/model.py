import torch
import torch.nn as nn
    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        RNN模型
        :param input_size: 输入特征的维度
        :param hidden_size: 隐藏层的大小
        :param output_size: 输出特征的维度
        :param num_layers: RNN层数
        """
        super(RNNModel, self).__init__()
        # 定义RNN层，num_layers表示RNN的层数
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 线性层，输出最后一个时间步的结果

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据 (batch_size, time_steps, input_size)
        :return: 预测结果
        """
        out, _ = self.rnn(x)  # out.shape = (batch_size, time_steps, hidden_size)
        out = out[:, -1, :]  # 只取最后一个时间步的输出
        out = self.fc(out)  # 将最后的输出通过全连接层得到最终的预测结果
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        LSTM模型
        :param input_size: 输入特征的维度
        :param hidden_size: 隐藏层的大小
        :param output_size: 输出特征的维度
        :param num_layers: LSTM层数
        """
        super(LSTMModel, self).__init__()
        # 定义LSTM层，num_layers表示LSTM的层数
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 线性层，输出最后一个时间步的结果

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据 (batch_size, time_steps, input_size)
        :return: 预测结果
        """
        out, (hn, cn) = self.lstm(x)  # out.shape = (batch_size, time_steps, hidden_size)
        out = out[:, -1, :]  # 只取最后一个时间步的输出
        out = self.fc(out)  # 将最后的输出通过全连接层得到最终的预测结果
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        GRU模型
        :param input_size: 输入特征的维度
        :param hidden_size: 隐藏层的大小
        :param output_size: 输出特征的维度
        :param num_layers: GRU层数
        """
        super(GRUModel, self).__init__()
        # 定义GRU层，num_layers表示GRU的层数
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 线性层，输出最后一个时间步的结果

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据 (batch_size, time_steps, input_size)
        :return: 预测结果
        """
        out, _ = self.gru(x)  # out.shape = (batch_size, time_steps, hidden_size)
        out = out[:, -1, :]  # 只取最后一个时间步的输出
        out = self.fc(out)  # 将最后的输出通过全连接层得到最终的预测结果
        return out
