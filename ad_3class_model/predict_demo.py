import torch
import os
import numpy as np
from train_lstm import MedicalLSTM

DATA_DIR = r"D:\projects\Python 运行环境\med-clf\processed_data"

def demo_predict(task="Na", window_hours=12, sample_idx=0):
    prefix = f"{task}_{window_hours}h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = MedicalLSTM(input_size=10, num_classes=3).to(device)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, f"best_{prefix}.pth")))
    model.eval()
    
    # 2. 从测试集里拿出一个真实病人的数据
    X_test = torch.load(os.path.join(DATA_DIR, f"{prefix}_test_X.pt"))
    M_test = torch.load(os.path.join(DATA_DIR, f"{prefix}_test_M.pt"))
    Y_test = torch.load(os.path.join(DATA_DIR, f"{prefix}_test_Y.pt"))
    
    # 提取第 sample_idx 个样本，并增加一个 Batch 维度 [1, 10, 12]
    x_sample = X_test[sample_idx:sample_idx+1].to(device)
    m_sample = M_test[sample_idx:sample_idx+1].to(device)
    y_true = Y_test[sample_idx].item()
    
    # 3. 让模型进行预测
    with torch.no_grad():
        outputs = model(x_sample, m_sample)
        # 使用 softmax 将输出转换为概率（三类概率之和为 1）
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
    # 获取概率最大的索引作为最终预测类别
    pred_class = np.argmax(probs)
    
    class_names = {0: '低(0)', 1: '正常(1)', 2: '高(2)'}
    
    print(f"\n🏥 === 医疗模型单条预测演示 ({task} {window_hours}h) ===")
    print(f"患者特征输入形状: {x_sample.shape} (1个样本, 10个特征, {window_hours}个时间步)")
    print(f"掩码矩阵有效观测点数: {int(m_sample.sum().item())} 个")
    print("-" * 40)
    print(f"模型预测概率分布:")
    print(f"  -> 低异常(0) 概率: {probs[0]:.2%}")
    print(f"  -> 正常(1) 概率:   {probs[1]:.2%}")
    print(f"  -> 高异常(2) 概率: {probs[2]:.2%}")
    print("-" * 40)
    print(f"👉 模型最终预测结论: 【{class_names[pred_class]}】")
    print(f"👉 该患者真实的医学标签: 【{class_names[y_true]}】")

if __name__ == "__main__":
    demo_predict(task="Na", window_hours=12, sample_idx=0)