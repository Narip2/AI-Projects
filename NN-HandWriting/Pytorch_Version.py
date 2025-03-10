import torch
import torch.nn as nn
import torch.optim as optim
import mnist_loader
import numpy as np

# 1. 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        input_size = 784
        hidden_size = 100
        output_size = 10
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 2. 创建模型实例
model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器

# 4. 获取数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
len_training_data = len(training_data)
X = np.array([x for x,y in training_data])
X = X.reshape(len_training_data,784)
X = torch.from_numpy(X).float()
Y = np.array([y for x,y in training_data])
Y = Y.reshape(len_training_data,10)
Y = torch.from_numpy(Y).float()

# 5. 训练循环
for epoch in range(1000):  # 训练 100 轮
    optimizer.zero_grad()  # 清空之前的梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    # 每 10 轮输出一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# verify the training model using test data
test_data = list(test_data)
len_test_data = len(test_data)

test_results = [(torch.argmax(model(torch.from_numpy(x.reshape(1,784)).float())), y) for x,y in test_data]
print(sum(int(x==y) for x,y in test_results))