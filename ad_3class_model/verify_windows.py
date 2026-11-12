import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from datetime import timedelta
import matplotlib.pyplot as plt

# 导入之前的模型类
from train_lstm import MedicalLSTM

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\raw_data"
PROCESSED_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FEATURE_NAMES = ['potassium', 'chloride', 'previous_sodium', 'glucose', 
                 'magnesium', 'urea_nitrogen', 'calcium_total', 'bicarbonate', 
                 'saline_volume_ml_proxy', 'fluid_net_ml_proxy']
SKEWED_FEATURES = ['glucose', 'urea_nitrogen', 'saline_volume_ml_proxy', 'fluid_net_ml_proxy']

# ================= 1. 生成特定窗口的数据 =================
def build_data_for_window(w):
    print(f"  -> 正在构建 {w}h 数据...")
    df_target = pd.read_csv(os.path.join(DATA_DIR, "03_血钠.csv"), encoding='utf-8-sig')
    df_target['event_time'] = pd.to_datetime(df_target['event_time'])
    df_target['label'] = df_target['value'].apply(lambda v: 0 if v < 135 else (1 if v <= 145 else 2))
    
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f[0].isdigit()]
    df_features = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f), encoding='utf-8-sig') for f in all_files], ignore_index=True)
    df_features = df_features[df_features['has_model_window'].astype(str).str.lower().isin(['true', '1'])]
    df_features['event_time'] = pd.to_datetime(df_features['event_time'])
    df_features['available_time'] = pd.to_datetime(df_features['available_time']).fillna(df_features['event_time'])

    X_list, Mask_list, Y_list, Subject_list = [], [], [], []
    
    # 抽样 3000 个目标点加快速度，保证验证流程快速跑完
    df_target_sample = df_target.sample(n=min(3000, len(df_target)), random_state=42)
    
    for _, row in df_target_sample.iterrows():
        t_time = row['event_time']
        start_time = t_time - timedelta(hours=w)
        window_data = df_features[(df_features['stay_id'] == row['stay_id']) & 
                                  (df_features['event_time'] >= start_time) & 
                                  (df_features['event_time'] < t_time) &
                                  (df_features['available_time'] < t_time)]
        
        if window_data.empty: continue
        x_t = np.full((len(FEATURE_NAMES), w), np.nan)
        m_t = np.zeros((len(FEATURE_NAMES), w))
        for _, f_row in window_data.iterrows():
            if f_row['feature'] in FEATURE_NAMES:
                i = FEATURE_NAMES.index(f_row['feature'])
                t_idx = int(np.floor((f_row['event_time'] - start_time).total_seconds() / 3600))
                if 0 <= t_idx < w:
                    x_t[i, t_idx] = f_row['value']
                    m_t[i, t_idx] = 1.0
        if m_t.sum() < 2: continue
        X_list.append(x_t); Mask_list.append(m_t); Y_list.append(row['label']); Subject_list.append(row['subject_id'])
        
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    M = torch.tensor(np.array(Mask_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.long)
    S = np.array(Subject_list)
    return X, M, Y, S

# ================= 2. 训练与评估 =================
# ================= 2. 训练与评估 (全类别版) =================
def train_and_eval(w, X, M, Y, S):
    print(f"  -> 正在训练并评估 {w}h 模型...")
    
    # 标准化防 NaN
    for i, feat_name in enumerate(FEATURE_NAMES):
        if feat_name in SKEWED_FEATURES:
            X[:, i, :] = torch.sign(X[:, i, :]) * torch.log1p(torch.abs(X[:, i, :]))
    for i in range(len(FEATURE_NAMES)):
        valid_idx = (M[:, i, :] == 1)
        if valid_idx.sum() > 0:
            mean = X[:, i, :][valid_idx].mean()
            std = X[:, i, :][valid_idx].std() + 1e-8
            X[:, i, :] = torch.where(valid_idx, (X[:, i, :] - mean) / std, torch.tensor(0.0))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, Y, groups=S))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalLSTM(input_size=10, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 2.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(TensorDataset(X[train_idx], M[train_idx], Y[train_idx]), batch_size=64, shuffle=True)
    for _ in range(10):
        model.train()
        for bx, bm, by in train_loader:
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx, bm), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    test_loader = DataLoader(TensorDataset(X[test_idx], M[test_idx], Y[test_idx]), batch_size=64)
    preds, probs, targets = [], [], []
    with torch.no_grad():
        for bx, bm, by in test_loader:
            out = model(bx.to(device), bm.to(device))
            probs.extend(torch.softmax(out, dim=1).cpu().numpy())
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(by.numpy())
            
    # 计算全类别的 Recall
    recalls = recall_score(targets, preds, labels=[0, 1, 2], average=None, zero_division=0)
    
    # 计算 Macro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    
    # 计算高钠 AUC
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(targets, classes=[0, 1, 2])
    try:
        auc_high = roc_auc_score(y_bin[:, 2], np.array(probs)[:, 2])
    except:
        auc_high = 0.5
        
    return recalls[0], recalls[1], recalls[2], macro_f1, auc_high

# ================= 3. 主流程 =================
if __name__ == "__main__":
    windows = [12, 24, 48, 72]
    results = []
    
    for w in windows:
        X, M, Y, S = build_data_for_window(w)
        if len(X) > 0:
            rec_low, rec_norm, rec_high, macro_f1, auc_high = train_and_eval(w, X, M, Y, S)
            results.append({
                '窗口': f"{w}h", 
                '低钠Recall': rec_low, 
                '正常Recall': rec_norm,
                '高钠Recall': rec_high, 
                'Macro_F1': macro_f1,
                '高钠AUC': auc_high
            })
        
    res_df = pd.DataFrame(results)
    print("\n=== 时间窗口验证结果 (全类别) ===")
    print(res_df.to_string(index=False))
    
    # 画图 (把低、正常、高的 Recall 都画出来)
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(res_df['窗口'], res_df['低钠Recall'], 'o-', label='低钠 Recall (类别0)')
        plt.plot(res_df['窗口'], res_df['正常Recall'], 's-', label='正常 Recall (类别1)')
        plt.plot(res_df['窗口'], res_df['高钠Recall'], '^-', label='高钠 Recall (类别2)')
        plt.plot(res_df['窗口'], res_df['Macro_F1'], 'k--', linewidth=2, label='Macro F1 (综合)')
        
        plt.xlabel('时间窗口')
        plt.ylabel('得分 (Score)')
        plt.title('不同时间窗口下血钠三分类性能全面对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DIR, "window_verify_all_classes.png"), dpi=300)
        print("\n✅ 验证完成！多类别图表已保存至 processed_data/window_verify_all_classes.png")
        
# ================= 3. 主流程 =================
if __name__ == "__main__":
    windows = [12, 24, 48, 72]
    results = []
    
    for w in windows:
        X, M, Y, S = build_data_for_window(w)
        if len(X) > 0:
            acc, rec, auc = train_and_eval(w, X, M, Y, S)
            results.append({'窗口': f"{w}h", 'Accuracy': acc, '高钠Recall': rec, '高钠AUC': auc})
        else:
            print(f"警告：{w}h 窗口未构建出有效样本！")
        
    res_df = pd.DataFrame(results)
    print("\n=== 时间窗口验证结果 ===")
    print(res_df.to_string(index=False))
    
    # 画图
    if not res_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        ax1.plot(res_df['窗口'], res_df['高钠Recall'], 'o-', color='red', label='高钠 Recall', linewidth=2)
        ax2.plot(res_df['窗口'], res_df['高钠AUC'], 's--', color='blue', label='高钠 AUC', linewidth=2)
        
        ax1.set_xlabel('时间窗口')
        ax1.set_ylabel('高钠 Recall (召回率)', color='red')
        ax2.set_ylabel('高钠 AUC', color='blue')
        plt.title('不同时间窗口下血钠预测性能对比')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DIR, "window_verify_result.png"), dpi=300)
        print("\n✅ 验证完成！图表已保存至 processed_data/window_verify_result.png")