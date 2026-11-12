import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import time

# ================= 1. 定义带 Mask 机制的 LSTM 模型 =================
class MedicalLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_classes=3, num_layers=2):
        super(MedicalLSTM, self).__init__()
        # 使用双向 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 双向的话，特征维度翻倍
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, mask):
        # x 和 mask 原始形状: [batch, 10, 12]
        
        # 1. 将 mask 转置为 [batch, 12, 10]，并压缩为时间步级别的掩码 [batch, 12, 1]
        # 逻辑：只要该时间步有任意一个特征被观测到（mask求和>0），该时间步就有效
        mask_transposed = mask.transpose(1, 2) # [batch, 12, 10]
        time_mask = (torch.sum(mask_transposed, dim=2) > 0).float().unsqueeze(-1) # [batch, 12, 1]
        
        # 2. 输入 LSTM
        x = x.transpose(1, 2) # [batch, 12, 10]
        out, _ = self.lstm(x) # out shape: [batch, 12, 128]
        
        # 3. 核心：利用时间步掩码屏蔽没有观测数据的时刻
        out = out * time_mask 
        
        # 4. 掩码平均池化：只对真实的观测时间步求平均
        sum_out = torch.sum(out, dim=1) # [batch, 128]
        sum_mask = torch.sum(time_mask, dim=1) + 1e-8 # [batch, 1] (加极小值防除以0)
        pooled_out = sum_out / sum_mask
        
        # 5. 分类输出
        logits = self.fc(pooled_out)
        return logits
    
# ================= 2. 训练流程 =================
def train_task(task="Na", window_hours=12, epochs=30):
    print(f"\n🚀 ========================================")
    print(f"🚀 开始训练模型: 任务【{task}】 | 窗口 {window_hours}h")
    print(f"🚀 ========================================")
    
    DATA_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data"
    prefix = f"{task}_{window_hours}h"
    
    try:
        X_train = torch.load(os.path.join(DATA_DIR, f"{prefix}_train_X.pt"))
        M_train = torch.load(os.path.join(DATA_DIR, f"{prefix}_train_M.pt"))
        Y_train = torch.load(os.path.join(DATA_DIR, f"{prefix}_train_Y.pt"))
        
        X_val = torch.load(os.path.join(DATA_DIR, f"{prefix}_val_X.pt"))
        M_val = torch.load(os.path.join(DATA_DIR, f"{prefix}_val_M.pt"))
        Y_val = torch.load(os.path.join(DATA_DIR, f"{prefix}_val_Y.pt"))
    except FileNotFoundError:
        print(f"❌ 找不到 {prefix} 的数据文件，请先确保预处理代码已跑完！")
        return

    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, M_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, M_val, Y_val), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")
    
    model = MedicalLSTM(input_size=10, num_classes=3).to(device)
    
    # 【核心武器】：加权交叉熵，惩罚少数类（假设正常是1，异常是0和2）
    # 权重设为 [5.0, 1.0, 10.0] (高钾极少，权重给到10)
    if task == "K":
        class_weights = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([3.0, 1.0, 2.0], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience = 5 # 早停：如果验证集 Loss 连续 5 个 epoch 不下降，就停止训练
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_m, batch_y in train_loader:
            batch_x, batch_m, batch_y = batch_x.to(device), batch_m.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, batch_m)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_m, batch_y in val_loader:
                batch_x, batch_m, batch_y = batch_x.to(device), batch_m.to(device), batch_y.to(device)
                outputs = model(batch_x, batch_m)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        train_loss_avg = total_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(DATA_DIR, f"best_{prefix}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 触发早停机制，在 Epoch {epoch+1} 停止训练。")
                break

    print(f"✅ 任务【{task}】 {window_hours}h 训练完成！最佳模型已保存为 best_{prefix}.pth")

# ================= 3. 主程序 =================
if __name__ == "__main__":
    # 咱们先跑一个血钠 12h 的模型看看效果
    train_task(task="Na", window_hours=12)
    train_task(task="Na", window_hours=24)
    train_task(task="K", window_hours=12)
    train_task(task="K", window_hours=24)