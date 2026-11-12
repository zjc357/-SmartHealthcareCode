import pandas as pd
import numpy as np
import os
import torch
from datetime import timedelta
from sklearn.model_selection import GroupShuffleSplit

# ================= 配置路径 =================
DATA_DIR = r"D:\projects\Python 运行环境\med-clf\raw_data"
OUTPUT_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特征列表（固定顺序）
FEATURE_NAMES = ['potassium', 'chloride', 'previous_sodium', 'glucose', 
                 'magnesium', 'urea_nitrogen', 'calcium_total', 'bicarbonate', 
                 'saline_volume_ml_proxy', 'fluid_net_ml_proxy']

# 极度右偏、需要 Log1p 变换的特征
SKEWED_FEATURES = ['glucose', 'urea_nitrogen', 'saline_volume_ml_proxy', 'fluid_net_ml_proxy']

def load_and_clean_data():
    print("1. 正在加载全量数据...")
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f[0].isdigit()]
    df_features = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f), encoding='utf-8-sig') for f in all_files], ignore_index=True)
    
    # 剔除无效窗口
    df_features = df_features[df_features['has_model_window'].astype(str).str.lower().isin(['true', '1'])]
    df_features['event_time'] = pd.to_datetime(df_features['event_time'])
    
    # 防泄露：如果 available_time 为空，默认用 event_time
    df_features['available_time'] = pd.to_datetime(df_features['available_time'])
    df_features['available_time'] = df_features['available_time'].fillna(df_features['event_time'])
    
    return df_features

def build_tensors_for_task(target_name, target_file, thresholds, window_hours, df_features):
    print(f"\n---> 正在处理任务: 预测【{target_name}】 | 窗口: {window_hours}h")
    
    # 读取目标数据
    df_target = pd.read_csv(os.path.join(DATA_DIR, target_file), encoding='utf-8-sig')
    df_target['event_time'] = pd.to_datetime(df_target['event_time'])
    
    # 阶段二：生成标签
    low_th, high_th = thresholds
    df_target['label'] = df_target['value'].apply(lambda v: 0 if v < low_th else (1 if v <= high_th else 2))
    
    # 阶段三 & 四：事件驱动的时序切片
    X_list, Mask_list, Y_list, Subject_list = [], [], [], []
    
    for idx, row in df_target.iterrows():
        t_time = row['event_time']
        t_label = row['label']
        stay_id = row['stay_id']
        subject_id = row['subject_id']
        
        start_time = t_time - timedelta(hours=window_hours)
        
        # 截取 [T-window, T) 窗口内且 available_time < T 的数据
        window_data = df_features[
            (df_features['stay_id'] == stay_id) & 
            (df_features['event_time'] >= start_time) & 
            (df_features['event_time'] < t_time) &
            (df_features['available_time'] < t_time) # 严格防泄露
        ]
        
        if window_data.empty:
            continue
            
        x_tensor = np.full((len(FEATURE_NAMES), window_hours), np.nan)
        mask_tensor = np.zeros((len(FEATURE_NAMES), window_hours))
        
        for _, f_row in window_data.iterrows():
            if f_row['feature'] in FEATURE_NAMES:
                feat_idx = FEATURE_NAMES.index(f_row['feature'])
                time_diff = (f_row['event_time'] - start_time).total_seconds() / 3600
                time_idx = int(np.floor(time_diff))
                
                if 0 <= time_idx < window_hours:
                    x_tensor[feat_idx, time_idx] = f_row['value']
                    mask_tensor[feat_idx, time_idx] = 1.0
                    
        X_list.append(x_tensor)
        Mask_list.append(mask_tensor)
        Y_list.append(t_label)
        Subject_list.append(subject_id)
        
    # 转换为 Tensor
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    M = torch.tensor(np.array(Mask_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.long)
    Subject_list = np.array(Subject_list)
    
    return X, M, Y, Subject_list

def split_and_scale(X, M, Y, Subjects, task_name, window_hours):
    # 阶段五：患者级划分 (70% 训练, 15% 验证, 15% 测试)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(X, Y, groups=Subjects))
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val.split(X[temp_idx], Y[temp_idx], groups=Subjects[temp_idx]))
    val_idx = temp_idx[val_idx] # 恢复全局索引
    test_idx = temp_idx[test_idx]
    
    print(f"    数据划分: 训练集 {len(train_idx)} | 验证集 {len(val_idx)} | 测试集 {len(test_idx)}")
    
    # 阶段四：Log1p 处理长尾极值 (针对特定特征)
    for i, feat_name in enumerate(FEATURE_NAMES):
        if feat_name in SKEWED_FEATURES:
            # 使用 np.sign(x) * np.log1p(np.abs(x)) 以兼容负数（如液体净值）
            X[:, i, :] = torch.sign(X[:, i, :]) * torch.log1p(torch.abs(X[:, i, :]))

    # 阶段五：标准化 (仅在 Mask==1 的真实值上计算均值和方差)
    X_train = X[train_idx]
    M_train = M[train_idx]
    
    feature_means = []
    feature_stds = []
    
    for i in range(len(FEATURE_NAMES)):
        # 提取该特征所有真实观测值
        valid_vals = X_train[:, i, :][M_train[:, i, :] == 1.0].numpy()
        if len(valid_vals) > 0:
            feature_means.append(np.nanmean(valid_vals))
            feature_stds.append(np.nanstd(valid_vals) + 1e-8) # 加极小值防除以0
        else:
            feature_means.append(0.0)
            feature_stds.append(1.0)
            
    # 应用标准化
    for i in range(len(FEATURE_NAMES)):
        # 仅对 Mask==1 的地方进行标准化，保留 NaN（或者填 0，由于有 Mask 在，填 0 不影响模型）
        mask_idx = (M[:, i, :] == 1.0)
        X[:, i, :] = torch.where(mask_idx, (X[:, i, :] - feature_means[i]) / feature_stds[i], torch.tensor(0.0))
        
    # 保存最终产物
    prefix = f"{task_name}_{window_hours}h"
    torch.save(X[train_idx], os.path.join(OUTPUT_DIR, f"{prefix}_train_X.pt"))
    torch.save(M[train_idx], os.path.join(OUTPUT_DIR, f"{prefix}_train_M.pt"))
    torch.save(Y[train_idx], os.path.join(OUTPUT_DIR, f"{prefix}_train_Y.pt"))
    
    torch.save(X[val_idx], os.path.join(OUTPUT_DIR, f"{prefix}_val_X.pt"))
    torch.save(M[val_idx], os.path.join(OUTPUT_DIR, f"{prefix}_val_M.pt"))
    torch.save(Y[val_idx], os.path.join(OUTPUT_DIR, f"{prefix}_val_Y.pt"))
    
    torch.save(X[test_idx], os.path.join(OUTPUT_DIR, f"{prefix}_test_X.pt"))
    torch.save(M[test_idx], os.path.join(OUTPUT_DIR, f"{prefix}_test_M.pt"))
    torch.save(Y[test_idx], os.path.join(OUTPUT_DIR, f"{prefix}_test_Y.pt"))

# ================= 主流程 =================
if __name__ == "__main__":
    df_features = load_and_clean_data()
    
    # 任务 A：血钠
    for w in [12, 24]:
        X, M, Y, S = build_tensors_for_task(
            target_name="Na", 
            target_file="03_血钠.csv", 
            thresholds=(135, 145), 
            window_hours=w, 
            df_features=df_features
        )
        split_and_scale(X, M, Y, S, "Na", w)
        
    # 任务 B：血钾
    for w in [12, 24]:
        X, M, Y, S = build_tensors_for_task(
            target_name="K", 
            target_file="01_血钾.csv", 
            thresholds=(3.5, 5.5), 
            window_hours=w, 
            df_features=df_features
        )
        split_and_scale(X, M, Y, S, "K", w)

    print("\n✅ 所有张量构建完毕！请在 processed_data 文件夹中查看 12 个 .pt 文件。")