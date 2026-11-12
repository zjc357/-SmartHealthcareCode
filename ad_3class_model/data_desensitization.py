import pandas as pd
import hashlib
import numpy as np
import glob
import os

# ================= 1. 路径配置 =================
# 原始数据路径（只读）
RAW_DATA_DIR = r"D:\projects\Python 运行环境\med-clf\raw_data"
# 脱敏后数据的输出路径（新建文件夹）
OUTPUT_DIR = r"D:\projects\Python 运行环境\med-clf\deidentified_data"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 辅助函数 =================
def hash_id(id_val):
    """使用加盐的 SHA-256 对 ID 进行哈希，截取前 12 位"""
    salt = "My_Secure_Medical_Salt_2026"
    return hashlib.sha256(f"{salt}{id_val}".encode()).hexdigest()[:12]

# ================= 3. 核心脱敏逻辑 =================
print("开始进行医疗数据脱敏...")

# 获取所有以数字开头的特征 CSV 文件
all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
feature_files = [f for f in all_files if os.path.basename(f)[0].isdigit()]

for file_path in feature_files:
    file_name = os.path.basename(file_path)
    print(f"正在处理: {file_name}")
    
    # 读取数据（严格使用 utf-8-sig 防止乱码）
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # --- 步骤 1：标识符哈希化 ---
    # 注意：因为每次哈希结果一样，所以同一个患者的记录依然能关联上
    if 'subject_id' in df.columns:
        df['subject_id'] = df['subject_id'].apply(hash_id)
    if 'hadm_id' in df.columns:
        df['hadm_id'] = df['hadm_id'].apply(hash_id)
    if 'stay_id' in df.columns:
        df['stay_id'] = df['stay_id'].apply(hash_id)
    
    # --- 步骤 2：时间平移（Date Shifting） ---
    if 'stay_id' in df.columns:
        # 为每个患者（stay_id）生成一个随机的天数偏移量（1 到 365 天）
        unique_stays = df['stay_id'].unique()
        time_shifts = {stay: np.random.randint(1, 365) for stay in unique_stays}
        
        # 定义需要平移的三个时间列
        time_cols = ['event_time', 'available_time', 'support_start']
        
        for col in time_cols:
            if col in df.columns:
                # 转换为时间格式，遇到空值强行转为 NaT (Not a Time)
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # 映射每个 stay_id 对应的偏移天数，并加到时间上
                shift_days = df['stay_id'].map(time_shifts)
                df[col] = df[col] + pd.to_timedelta(shift_days, unit='D')
                
    # --- 步骤 3：剔除无用字段 ---
    if 'has_model_window' in df.columns:
        df = df.drop(columns=['has_model_window'])
        
    # --- 步骤 4：保存脱敏数据 ---
    out_path = os.path.join(OUTPUT_DIR, file_name)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

print("\n✅ 所有数据脱敏完成！")
print(f"脱敏后的文件已安全保存在: {OUTPUT_DIR}")
print("原始 raw_data 文件夹中的数据未受任何修改。")