import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "report_figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

features = ['potassium', 'chloride', 'previous_sodium', 'glucose', 'magnesium', 
            'urea_nitrogen', 'calcium_total', 'bicarbonate', 'saline_volume_ml_proxy', 'fluid_net_ml_proxy']

def analyze_coverage(task, window):
    print(f"正在分析 {task} {window}h 的覆盖率...")
    M = torch.load(os.path.join(DATA_DIR, f"{task}_{window}h_train_M.pt"))
    
    # 1. 统计所有样本中，每个特征的平均观测次数 (每个样本有 window 个时间步)
    # M shape: [样本数, 10, window]
    avg_observations_per_feature = M.sum(dim=(0, 2)).numpy() / M.shape[0]
    
    # 2. 统计全空样本的比例（即整个样本的所有 10 个特征在窗口内全为 0）
    total_obs_per_sample = M.sum(dim=(1, 2)).numpy()
    empty_samples = np.sum(total_obs_per_sample == 0) / len(total_obs_per_sample) * 100
    
    print(f"  - 全空样本比例: {empty_samples:.2f}%")
    print(f"  - 各特征平均每个样本的观测次数: {dict(zip(features, np.round(avg_observations_per_feature, 2)))}")
    
    return avg_observations_per_feature

# 对比 12h 和 24h
avg_12 = analyze_coverage("Na", 12)
avg_24 = analyze_coverage("Na", 24)

# 画对比图
x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width/2, avg_12, width, label='12小时窗口', color='skyblue')
rects2 = ax.bar(x + width/2, avg_24, width, label='24小时窗口', color='coral')

ax.set_ylabel('平均观测次数 (次/样本)')
ax.set_title('12h 与 24h 窗口下各特征的平均观测频次对比')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_coverage_12_vs_24.png"), dpi=300)
plt.close()

print("\n✅ 分析完成，请查看生成的图表。")