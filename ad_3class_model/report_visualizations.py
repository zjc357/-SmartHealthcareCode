import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\raw_data"
OUTPUT_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data\report_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 图1：目标变量分布图 (证明类别不平衡) =================
print("正在生成标签分布图...")
df_na = pd.read_csv(os.path.join(DATA_DIR, "03_血钠.csv"), encoding='utf-8-sig')
df_k = pd.read_csv(os.path.join(DATA_DIR, "01_血钾.csv"), encoding='utf-8-sig')

df_na['label'] = df_na['value'].apply(lambda v: 0 if v < 135 else (1 if v <= 145 else 2))
df_k['label'] = df_k['value'].apply(lambda v: 0 if v < 3.5 else (1 if v <= 5.5 else 2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 血钠
sns.countplot(data=df_na, x='label', ax=axes[0], palette='Blues')
axes[0].set_title("任务A：血钠 (Na+) 三分类标签分布", fontsize=14)
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['低钠 (<135)', '正常 (135-145)', '高钠 (>145)'])
axes[0].set_ylabel("样本数量")

# 血钾
sns.countplot(data=df_k, x='label', ax=axes[1], palette='Reds')
axes[1].set_title("任务B：血钾 (K+) 三分类标签分布", fontsize=14)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['低钾 (<3.5)', '正常 (3.5-5.5)', '高钾 (>5.5)'])
axes[1].set_ylabel("样本数量")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_label_distribution.png"), dpi=300)
plt.close()

# ================= 图2：12h 时序切片与 Mask 示意图 =================
print("正在生成 12h 窗口构造示意图...")
# 随机挑一个预处理好的样本
X_sample = torch.load(os.path.join(r"D:\projects\Python 运行环境\med-clf\processed_data", "Na_12h_train_X.pt"))[0].numpy()
M_sample = torch.load(os.path.join(r"D:\projects\Python 运行环境\med-clf\processed_data", "Na_12h_train_M.pt"))[0].numpy()

features = ['potassium', 'chloride', 'previous_sodium', 'glucose', 'magnesium', 
            'urea_nitrogen', 'calcium_total', 'bicarbonate', 'saline_volume_ml_proxy', 'fluid_net_ml_proxy']

plt.figure(figsize=(10, 6))
# 画 Mask 热力图，1 表示有数据，0 表示缺失
sns.heatmap(M_sample, cmap="YlGnBu", cbar=False, 
            xticklabels=[f"{i}h" for i in range(12)], 
            yticklabels=features)
plt.title("样本示例：前 12 小时 10 项特征的真实观测分布 (Mask矩阵)", fontsize=14)
plt.xlabel("时间步 (从前12小时至目标点T)")
plt.ylabel("临床特征")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_window_mask_example.png"), dpi=300)
plt.close()

print("✅ 图表已生成，保存在 processed_data/report_figs 文件夹。")
