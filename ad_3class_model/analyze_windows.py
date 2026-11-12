import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import seaborn as sns

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\raw_data"
OUTPUT_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data\report_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. 正在加载数据...")
df_target = pd.read_csv(os.path.join(DATA_DIR, "03_血钠.csv"), encoding='utf-8-sig')
df_target['event_time'] = pd.to_datetime(df_target['event_time'])

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f[0].isdigit()]
df_features = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f), encoding='utf-8-sig') for f in all_files], ignore_index=True)
df_features['event_time'] = pd.to_datetime(df_features['event_time'])
df_features['available_time'] = pd.to_datetime(df_features['available_time']).fillna(df_features['event_time'])

windows = [12, 24, 48, 72]
results = []

# 随机抽样 3000 个检测点作为分析对象（保证运行速度）
df_target_sample = df_target.sample(n=min(3000, len(df_target)), random_state=42)

print("2. 开始统计不同窗口的特征变化 (这可能需要几十秒)...")
for w in windows:
    print(f"正在分析 {w}h 窗口...")
    total_obs = 0
    valid_samples = 0
    glucose_std_list = []
    
    for idx, row in df_target_sample.iterrows():
        t_time = row['event_time']
        stay_id = row['stay_id']
        start_time = t_time - timedelta(hours=w)
        
        window_data = df_features[
            (df_features['stay_id'] == stay_id) & 
            (df_features['event_time'] >= start_time) & 
            (df_features['event_time'] < t_time) &
            (df_features['available_time'] < t_time)
        ]
        
        if not window_data.empty:
            valid_samples += 1
            total_obs += len(window_data)
            
            # 算一下葡萄糖的方差作为“生理波动/噪声”的代表
            g_vals = window_data[window_data['feature'] == 'glucose']['value']
            if len(g_vals) > 1:
                glucose_std_list.append(g_vals.std())
                
    avg_obs = total_obs / valid_samples if valid_samples > 0 else 0
    avg_glucose_std = np.mean(glucose_std_list) if glucose_std_list else 0
    
    results.append({
        '窗口大小': f"{w}h",
        '有数据的样本比例': valid_samples / len(df_target_sample) * 100,
        '平均特征观测数': avg_obs,
        '血糖波动标准差(噪声)': avg_glucose_std
    })

res_df = pd.DataFrame(results)
print("\n=== 不同时间窗口数据特征变化 ===")
print(res_df.to_string(index=False))

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：信息量（平均观测次数）
sns.barplot(data=res_df, x='窗口大小', y='平均特征观测数', ax=axes[0], palette='Blues')
axes[0].set_title("随窗口拉长：特征信息量变化 (平均观测次数)", fontsize=14)
axes[0].set_ylabel("平均记录条数 (次/样本)")

# 图2：噪声（血糖标准差）
sns.barplot(data=res_df, x='窗口大小', y='血糖波动标准差(噪声)', ax=axes[1], palette='Reds')
axes[1].set_title("随窗口拉长：数据噪声变化 (血糖历史波动)", fontsize=14)
axes[1].set_ylabel("标准差 (越大说明波动越剧烈)")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "window_analysis.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\n✅ 分析完成！图表已保存至: {plot_path}")