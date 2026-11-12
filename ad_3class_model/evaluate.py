import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 引入刚才的模型类
from train_lstm import MedicalLSTM

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "evaluation_figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(task="Na", window_hours=12):
    print(f"\n📊 正在评估模型: 任务【{task}】 | 窗口 {window_hours}h")
    prefix = f"{task}_{window_hours}h"
    
    # 加载测试集
    X_test = torch.load(os.path.join(DATA_DIR, f"{prefix}_test_X.pt"))
    M_test = torch.load(os.path.join(DATA_DIR, f"{prefix}_test_M.pt"))
    Y_test = torch.load(os.path.join(DATA_DIR, f"{prefix}_test_Y.pt"))
    
    test_loader = DataLoader(TensorDataset(X_test, M_test, Y_test), batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalLSTM(input_size=10, num_classes=3).to(device)
    
    # 加载最佳权重
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, f"best_{prefix}.pth")))
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_m, batch_y in test_loader:
            batch_x, batch_m = batch_x.to(device), batch_m.to(device)
            outputs = model(batch_x, batch_m)
            
            # 获取概率和预测类别
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # ================= 1. 输出分类报告 (Precision, Recall, F1) =================
    print("\n--- 分类报告 (Classification Report) ---")
    print(classification_report(all_targets, all_preds, target_names=['低(0)', '正常(1)', '高(2)']))
    
    # ================= 2. 画混淆矩阵 =================
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['低(0)', '正常(1)', '高(2)'], 
                yticklabels=['低(0)', '正常(1)', '高(2)'])
    plt.title(f'{task} {window_hours}h 混淆矩阵 (Confusion Matrix)')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # ================= 3. 画多分类 ROC 曲线 =================
    # 将标签二值化 (One-vs-Rest)
    y_test_bin = label_binarize(all_targets, classes=[0, 1, 2])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    labels = ['低(0)', '正常(1)', '高(2)']
    
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC 曲线 {labels[i]} (AUC = {roc_auc[i]:.4f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title(f'{task} {window_hours}h 三分类 ROC 曲线')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_roc_curve.png"), dpi=300)
    plt.close()

    print(f"✅ 评估完成！图表已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate_model(task="Na", window_hours=12)
    evaluate_model(task="Na", window_hours=24)  # 解开
    evaluate_model(task="K", window_hours=12)   # 解开
    evaluate_model(task="K", window_hours=24)   # 解开