import torch
from torch import nn
from model import resnet34
from torchvision import datasets, transforms
from torch import optim
import os

# 数据预处理
data_transform = transforms.Compose([
    transforms.ToTensor(),
    # 可以加入其他数据增强操作
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化模型
model = resnet34().to(device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    total_loss, total_acc, n = 0.0, 0.0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()  # 使用当前批次的损失
        optimizer.step()

        total_loss += cur_loss.item()
        total_acc += cur_acc.item()
        n += 1

    train_loss = total_loss / n
    train_acc = total_acc / n
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Train Accuracy: {train_acc:.4f}')

# 验证函数
def val(dataloader, model, loss_fn):
    total_loss, total_acc, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            total_loss += cur_loss.item()
            total_acc += cur_acc.item()
            n += 1

        val_loss = total_loss / n
        val_acc = total_acc / n
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        return val_acc

# 训练循环
num_epochs = 5
best_acc = 0
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    train(train_dataloader, model, loss_fn, optimizer)
    acc = val(test_dataloader, model, loss_fn)
    if acc > best_acc:
        best_acc = acc
        if not os.path.exists('save_model'):
            os.mkdir('save_model')
        print('Saving best model')
        torch.save(model.state_dict(), './save_model/ckpt.pth')

print('Finished Training')
