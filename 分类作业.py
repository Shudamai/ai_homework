# 导入必要的库
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ========== 配置设备（优先使用 GPU 或 Apple MPS） ==========
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ========== 图像预处理变换 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== 加载数据集 ==========
data_dir = r'/Users/shudamai/Downloads/catvsdog 2'

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== 加载预训练模型并修改输出层 ==========
from torchvision.models import resnet34, ResNet34_Weights
weights = ResNet34_Weights.DEFAULT
model = resnet34(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ========== 定义损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ========== 模型训练函数 ==========
def train(num_epochs=10):
    train_acc, val_acc = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            running_corrects += torch.sum(preds == labels)

        epoch_train_acc = running_corrects.float() / len(train_dataset)
        train_acc.append(epoch_train_acc.item())

        # 验证阶段
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                val_corrects += torch.sum(preds == labels)

        epoch_val_acc = val_corrects.float() / len(val_dataset)
        val_acc.append(epoch_val_acc.item())

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")

    return train_acc, val_acc

# ========== 执行训练 ==========
train_acc, val_acc = train(num_epochs=10)

# ========== 绘制准确率曲线 ==========
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# ========== 加载最佳模型并进行测试 ==========
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, 1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 构建预测结果 DataFrame
image_paths = [path for path, _ in test_dataset.samples]
results_df = pd.DataFrame({
    "image_path": image_paths,
    "true_label": [test_dataset.classes[i] for i in all_labels],
    "predicted_label": [test_dataset.classes[i] for i in all_preds]
})
results_df.to_csv("test_predictions.csv", index=False)
print("✅ 测试集分类结果已保存到 test_predictions.csv")

# ========== 绘制混淆矩阵 ==========
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on Test Set")
plt.show()
