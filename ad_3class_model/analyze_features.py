import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\raw_data"
OUTPUT_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data\report_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("1. 加载数据...")
df_target = pd.read_csv(os.path.join(DATA_DIR, "03_血钠.csv"), encoding='utf-8-sig')
df_target['event_time'] = pd.to_datetime(df_target['event_time'])

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f[0].isdigit()]
df_features = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f), encoding='utf-8-sig') for f in all_files], ignore_index=True)
df_features['event_time'] = pd.to_datetime(df_features['event_time'])
df_features['available_time'] = pd.to_datetime(df_features['available_time']).fillna(df_features['event_time'])

windows = [12, 24, 48, 72]
results = []

# 抽样3000个点，聚焦分析“血钠（previous_sodium）”这个最重要的特征
df_target_sample = df_target.sample(n=min(3000, len(df_target)), random_state=42)

print("2. 分析不同时间窗口的特征变化 (聚焦核心特征: 血钠)...")
for w in windows:
    obs_counts = []
    lags = [] # 数据滞后时间（目标时间 - 记录发生时间）
    variances = [] # 历史血钠的波动幅度
    
    for _, row in df_target_sample.iterrows():
        t_time = row['event_time']
        start_time = t_time - timedelta(hours=w)
        
        # 提取窗口内的血钠特征
        window_data = df_features[
            (df_features['stay_id'] == row['stay_id']) & 
            (df_features['feature'] == 'previous_sodium') &
            (df_features['event_time'] >= start_time) & 
            (df_features['event_time'] < t_time) &
            (df_features['available_time'] < t_time)
        ]
        
        if not window_data.empty:
            obs_counts.append(len(window_data))
            # 计算该窗口内所有记录的滞后小时数
            lags.extend(((t_time - window_data['event_time']).dt.total_seconds() / 3600).tolist())
            # 计算该窗口内血钠值的方差（波动程度）
            if len(window_data) > 1:
                variances.append(window_data['value'].std())
                
    avg_count = np.mean(obs_counts) if obs_counts else 0
    # 平均滞后：比如12h窗口，最近一次抽血可能在2小时前，最远在12小时前，求个平均
    mean_lag = np.mean(lags) if lags else w 
    # 用中位数表示大多数数据有多“老”
    median_lag = np.median(lags) if lags else w
    avg_var = np.mean(variances) if variances else 0
    
    results.append({
        '窗口': f"{w}h",
        '平均观测次数': avg_count,
        '平均滞后时间(h)': mean_lag,
        '中位数滞后时间(h)': median_lag,
        '历史值波动(标准差)': avg_var
    })

res_df = pd.DataFrame(results)
print("\n=== 特征动态变化分析结果 ===")
print(res_df.to_string(index=False))

# 画图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1：观测次数
sns.barplot(data=res_df, x='窗口', y='平均观测次数', ax=axes[0], color='skyblue')
axes[0].set_title("信息量变化：窗口内血钠平均观测次数")
axes[0].set_ylabel("次数")

# 图2：数据滞后时间（陈旧度）
sns.barplot(data=res_df, x='窗口', y='中位数滞后时间(h)', ax=axes[1], color='salmon')
axes[1].set_title("时效性变化：数据的滞后时间中位数")
axes[1].set_ylabel("小时 (越大说明数据越老)")
# 添加辅助线：超过24小时的数据在临床上基本是“老黄历”
axes[1].axhline(y=24, color='red', linestyle='--', label='24小时陈旧度警戒线')
axes[1].legend()

# 图3：历史值波动
sns.barplot(data=res_df, x='窗口', y='历史值波动(标准差)', ax=axes[2], color='lightgreen')
axes[2].set_title("噪声变化：历史血钠波动幅度")
axes[2].set_ylabel("标准差 (越大说明患者状态越剧烈)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_dynamics.png"), dpi=300)
plt.close()

print(f"\n✅ 分析完成！图表已保存至 {OUTPUT_DIR}/feature_dynamics.png")