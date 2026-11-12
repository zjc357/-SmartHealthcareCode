# -*- coding: utf-8 -*-

"""
指南约束 + Clinical World Model 反事实推理脚本
（严格对标《SepsisAgent 指南约束模块：具体设计与实现方案》修订版）

流程：
    1. Propose: 使用 G01-G06 对 25 个治疗动作进行指南评估。
    2. Simulate: 使用训练好的 GRU Clinical World Model 对未发生明确规则冲突的动作进行下一状态预测。
    3. Refine: 基于模型不确定性、状态分布、生理范围和动作支持度拒绝不可信的反事实模拟。
    4. 展示: 移除复合打分公式，直接展示指南约束层级、预测死亡率、升压药需求概率。

相对上一版的修订（均为对齐审查中发现的偏差）：
    a. 修复 check_action_support：train_cwm.py 保存的 action_counts 是
       dict，原代码按数组索引会直接出错；现兼容 dict 与 list 两种格式，
       统一转换为长度 25 的计数向量。
    b. 修正"未发现明确规则冲突"的误导性输出：指南模块修订后
       RULE_CONFLICT 已可达（默认仅 G05 一条待审核冲突），统计与
       拒绝路径按真实数量报告。
    c. 明确 INSUFFICIENT_INFORMATION 动作的处置：按流程设计仅
       RULE_CONFLICT 动作不进入模拟；INSUFFICIENT 动作仍模拟但
       在排序中低于 CONDITIONAL / SUPPORTED（见 get_status_priority）。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. 路径配置
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = Path(r"C:\Users\恭喜发财\Desktop\data-1793668464627.csv")
DEFAULT_CHECKPOINT = CURRENT_DIR / "cwm_checkpoint.pt"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "outputs"
RULE_FILE = CURRENT_DIR / "完整流程代码.py"

# ============================================================
# 2. 动态加载指南约束模块
# ============================================================

if not RULE_FILE.exists():
    raise FileNotFoundError(f"未找到指南规则引擎：{RULE_FILE}")

spec = importlib.util.spec_from_file_location("guideline_rules", str(RULE_FILE))
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载指南规则引擎：{RULE_FILE}")

rules = importlib.util.module_from_spec(spec)
sys.modules["guideline_rules"] = rules
spec.loader.exec_module(rules)

# ============================================================
# 3. 默认状态变量与模型配置
# ============================================================

DEFAULT_STATE_VARS = [
    "hr", "sysbp", "diabp", "map", "rr", "temp", "spo2", "gcs_min", "fio2", "mech_vent", "urine_output",
    "lactate", "creatinine", "wbc", "platelet", "bilirubin_total", "ph", "bun", "sodium", "potassium",
    "chloride", "glucose", "albumin", "calcium_total", "calcium_free", "magnesium", "hemoglobin",
    "hematocrit", "inr", "pt", "ptt", "ast", "alt", "bicarbonate", "base_excess", "anion_gap",
    "pao2", "paco2", "total_co2", "age", "gender", "weight", "elixhauser_index", "icu_readmission",
]

SEQ_LEN = 12
NUM_ACTIONS = 25
ACTION_EMBED_DIM = 32
HIDDEN_DIM = 128


# ============================================================
# 4. 与训练代码一致的 CWM 模型
# ============================================================

class ClinicalWorldModel(nn.Module):
    def __init__(self, state_dim: int, num_actions: int = NUM_ACTIONS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, ACTION_EMBED_DIM)
        gru_input_dim = state_dim + state_dim + ACTION_EMBED_DIM
        self.gru = nn.GRU(input_size=gru_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, state_dim * 2))
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.vr_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, states: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_embedding = self.action_embed(actions)
        x = torch.cat([states, masks, action_embedding], dim=-1)
        output, _ = self.gru(x)
        hidden_last = output[:, -1, :]

        state_params = self.state_head(hidden_last)
        mu, logvar = state_params.chunk(2, dim=-1)

        mortality_logit = self.outcome_head(hidden_last).squeeze(-1)
        vasopressor_logit = self.vr_head(hidden_last).squeeze(-1)

        return mu, logvar, mortality_logit, vasopressor_logit


# ============================================================
# 5. 数据和 checkpoint 工具函数
# ============================================================

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到模型 checkpoint：{checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1)

def get_state_vars(checkpoint: Dict[str, Any]) -> List[str]:
    saved_state_vars = checkpoint.get("state_vars")
    if saved_state_vars is None:
        return list(DEFAULT_STATE_VARS)
    return [str(item) for item in list(saved_state_vars)]

def ensure_data_columns(df: pd.DataFrame, state_vars: List[str]) -> None:
    required_columns = ["stay_id", "hour_from_onset", "tev_ml", "ne_eq"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    missing_state_columns = [c for c in state_vars if c not in df.columns]
    if missing_columns:
        raise KeyError(f"CSV 缺少 CWM 所需字段：{', '.join(missing_columns)}")
    if missing_state_columns:
        raise KeyError(f"CSV 缺少 checkpoint 对应的状态字段：{', '.join(missing_state_columns)}")

def normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "gender" not in result.columns:
        return result
    if pd.api.types.is_numeric_dtype(result["gender"]):
        result["gender"] = pd.to_numeric(result["gender"], errors="coerce").fillna(0.0)
        return result
    mapping = {"M": 0.0, "F": 1.0, "Male": 0.0, "Female": 1.0, "male": 0.0, "female": 1.0}
    result["gender"] = result["gender"].map(mapping).fillna(0.0).astype(float)
    return result

def safe_array_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=np.float32).copy()
    denominator[np.abs(denominator) < 1e-8] = 1.0
    return numerator / denominator

def numeric_scalar(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


# ============================================================
# 6. 动作离散化
# ============================================================

def discretize_action_value(value: Any, quantiles: np.ndarray) -> int:
    numeric = numeric_scalar(value)
    if numeric is None or numeric <= 0:
        return 0
    for index, threshold in enumerate(quantiles.tolist()):
        if numeric <= float(threshold):
            return index + 1
    return 4

def build_action_ids(df: pd.DataFrame, tev_quantiles: np.ndarray, ne_quantiles: np.ndarray) -> pd.DataFrame:
    result = df.copy()
    result["tev_bin"] = result["tev_ml"].apply(lambda v: discretize_action_value(v, tev_quantiles))
    result["ne_bin"] = result["ne_eq"].apply(lambda v: discretize_action_value(v, ne_quantiles))
    result["action_id"] = result["tev_bin"] * 5 + result["ne_bin"]
    return result


# ============================================================
# 7. 历史序列构建 (验证 Mask 机制)
# ============================================================

def prepare_history(
    df: pd.DataFrame,
    current_index: int,
    state_vars: List[str],
    train_mean: np.ndarray,
    train_std: np.ndarray,
    tev_quantiles: np.ndarray,
    ne_quantiles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    history_start = max(0, current_index - SEQ_LEN + 1)
    history = df.iloc[history_start: current_index + 1].copy()
    history = build_action_ids(history, tev_quantiles, ne_quantiles)

    state_frame = history[state_vars].copy()

    # 【核心对齐】生成 Mask 矩阵，缺失值为 0，非缺失值为 1
    # 这与 train_cwm.py 中 group[STATE_VARS].notna().astype(np.float32) 完全一致
    mask = state_frame.notna().astype(np.float32).to_numpy()

    fill_values = pd.Series(train_mean, index=state_vars, dtype=np.float32)
    filled = state_frame.fillna(fill_values)
    state_values = filled.to_numpy(dtype=np.float32)
    normalized = safe_array_divide(state_values - train_mean, train_std)

    actions = history["action_id"].to_numpy(dtype=np.int64)
    pad_len = SEQ_LEN - len(history)

    if pad_len > 0:
        normalized = np.pad(normalized, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        mask = np.pad(mask, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        actions = np.pad(actions, (pad_len, 0), mode="constant", constant_values=0)

    current_raw_state = state_values[-1].copy()
    return normalized.astype(np.float32), mask.astype(np.float32), actions.astype(np.int64), current_raw_state.astype(np.float32)


# ============================================================
# 8. 可信度检查
# ============================================================

def get_state_index(state_vars: List[str], name: str) -> Optional[int]:
    try:
        return state_vars.index(name)
    except ValueError:
        return None

def check_state_distribution(current_raw_state: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray, max_z: float) -> Tuple[bool, str]:
    current_z = safe_array_divide(current_raw_state - train_mean, train_std)
    finite_z = current_z[np.isfinite(current_z)]
    if finite_z.size == 0:
        return False, "当前状态没有可用数值。"
    max_abs_z = float(np.max(np.abs(finite_z)))
    if max_abs_z > max_z:
        return False, f"当前状态偏离训练分布过大 (max_abs_z={max_abs_z:.3f}, threshold={max_z:.3f})"
    return True, f"当前状态分布检查通过 (max_abs_z={max_abs_z:.3f})"

def check_prediction_uncertainty(sigma_norm: np.ndarray, train_std: np.ndarray, max_mean_sigma_raw: float, max_sigma_z: float) -> Tuple[bool, str, float]:
    sigma_norm = np.asarray(sigma_norm, dtype=np.float32)
    sigma_raw = np.abs(train_std) * sigma_norm
    mean_sigma_raw = float(np.mean(sigma_raw))
    max_sigma = float(np.max(sigma_norm))

    if max_sigma > max_sigma_z:
        return False, f"标准化预测不确定性过大 (max_sigma_z={max_sigma:.3f})", mean_sigma_raw
    if mean_sigma_raw > max_mean_sigma_raw:
        return False, f"原始尺度平均预测标准差过大 (mean_sigma={mean_sigma_raw:.3f})", mean_sigma_raw
    return True, f"预测不确定性检查通过 (mean_sigma={mean_sigma_raw:.3f}, max_sigma_z={max_sigma:.3f})", mean_sigma_raw

def check_physiological_range(predicted_state: np.ndarray, state_vars: List[str]) -> Tuple[bool, str]:
    map_index = get_state_index(state_vars, "map")
    lactate_index = get_state_index(state_vars, "lactate")

    if map_index is not None:
        map_value = float(predicted_state[map_index])
        if not 20.0 <= map_value <= 200.0:
            return False, f"预测 MAP 超出生理检查范围：{map_value:.3f}"

    if lactate_index is not None:
        lactate_value = float(predicted_state[lactate_index])
        if not 0.0 <= lactate_value <= 30.0:
            return False, f"预测乳酸超出生理检查范围：{lactate_value:.3f}"

    return True, "预测结果生理范围检查通过。"

def check_action_support(action_id: int, checkpoint: Dict[str, Any], min_count: int) -> Tuple[bool, str, Optional[int]]:
    """
    修复说明：train_cwm.py 保存的 action_counts 是 {action_id: count} 字典，
    原实现直接 np.asarray(dict) 会得到 0 维对象数组，counts[action_id]
    必然出错。现兼容 dict 与 list/ndarray 两种格式，统一转换为
    长度 NUM_ACTIONS 的计数向量后再索引。
    """
    action_counts = checkpoint.get("action_counts")
    if action_counts is None:
        return True, "checkpoint 未保存 action_counts，动作覆盖度未评估。", None

    if isinstance(action_counts, dict):
        counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        for key, value in action_counts.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= index < NUM_ACTIONS:
                counts[index] = int(value)
    else:
        counts = np.asarray(action_counts, dtype=np.int64).reshape(-1)

    if action_id >= len(counts):
        return False, f"action_id={action_id} 超出 checkpoint 的动作统计范围。", None

    count = int(counts[action_id])
    if count < min_count:
        return False, f"动作训练支持度不足：count={count}, min_count={min_count}", count

    return True, f"动作训练支持度通过：count={count}", count


# ============================================================
# 9. 单患者推理主流程
# ============================================================

def run_pipeline(
    data_path: Path,
    checkpoint_path: Path,
    stay_id: Any,
    time_index: Optional[float],
    output_dir: Path,
    max_state_z: float,
    max_mean_sigma_raw: float,
    max_sigma_z: float,
    min_action_count: int,
) -> None:
    print("[1] 正在读取数据...")
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    if df.empty:
        raise ValueError("CSV 文件为空。")
    if "stay_id" not in df.columns:
        raise KeyError("CSV 中没有 stay_id 字段。")
    if stay_id is None:
        stay_id = df["stay_id"].iloc[0]

    patient_df = df[df["stay_id"].astype(str) == str(stay_id)].copy()
    if patient_df.empty:
        raise ValueError(f"没有找到 stay_id={stay_id}")

    patient_df["hour_from_onset"] = pd.to_numeric(patient_df["hour_from_onset"], errors="coerce")
    patient_df = patient_df.sort_values("hour_from_onset").reset_index(drop=True)

    if time_index is None:
        current_index = len(patient_df) - 1
    else:
        distance = (patient_df["hour_from_onset"] - float(time_index)).abs()
        current_index = int(distance.idxmin())

    current_time = patient_df.loc[current_index, "hour_from_onset"]
    print(f"    -> 正在推理患者：{stay_id}")
    print(f"[2] 提取当前状态 (时间点：{float(current_time):.1f}h)")

    print("[3] 运行 G01-G06 指南约束引擎...")
    detected = rules.detect_columns(patient_df)
    guideline_context = rules.build_guideline_context(
        trajectory=patient_df,
        detected=detected,
        decision_index=current_index,
    )
    all_actions = rules.evaluate_all_actions(guideline_context)

    status_counts = Counter(item["merged_status"] for item in all_actions)
    print("    -> 指南状态统计：" + json.dumps(dict(status_counts), ensure_ascii=False))

    # ==========================================================
    # 统计 INSUFFICIENT 具体原因（临床数据缺失分析）
    # ==========================================================
    insuff_g03_count = 0
    insuff_g05_count = 0
    for action in all_actions:
        if action["merged_status"] == rules.STATUS_INSUFFICIENT:
            for assessment in action["rule_assessments"]:
                if assessment["status"] == rules.STATUS_INSUFFICIENT:
                    if assessment["rule_id"] == "G03": insuff_g03_count += 1
                    if assessment["rule_id"] == "G05": insuff_g05_count += 1

    print(f"    -> [数据缺失审查] 25 个动作中：")
    print(f"       因 G03(液体反应性缺失) 标记为 INSUFFICIENT 的动作数：{insuff_g03_count}")
    print(f"       因 G05(容量负荷证据缺失) 标记为 INSUFFICIENT 的动作数：{insuff_g05_count}")

    # 修订说明：指南模块修订后 RULE_CONFLICT 已可达（默认仅 G05 一条
    # 待临床审核冲突），此处按真实数量报告，不再恒定输出 25。
    # 按流程设计，仅 RULE_CONFLICT 动作不进入 CWM 模拟；
    # INSUFFICIENT 动作仍会模拟，但在最终排序中处于最低优先级。
    candidates = [item for item in all_actions if item["merged_status"] != rules.STATUS_RULE_CONFLICT]
    conflict_count = 25 - len(candidates)
    print(f"    -> 规则引擎标记 RULE_CONFLICT 的动作：{conflict_count}/25（不进入模拟）")
    print(f"    -> 进入反事实模拟的动作：{len(candidates)}/25")
    if conflict_count > 0:
        print("       提示：RULE_CONFLICT 判定为保守候选，需经临床审核确认，详见规则引擎 RULE_CONFLICT_DEFINITIONS。")

    print("[4] 加载 Clinical World Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)

    state_vars = get_state_vars(checkpoint)
    ensure_data_columns(patient_df, state_vars)

    train_mean = to_numpy_1d(checkpoint["train_mean"])
    train_std = to_numpy_1d(checkpoint["train_std"])
    tev_quantiles = to_numpy_1d(checkpoint["tev_q"])
    ne_quantiles = to_numpy_1d(checkpoint["ne_q"])

    patient_df = normalize_gender(patient_df)
    patient_df = build_action_ids(patient_df, tev_quantiles, ne_quantiles)

    model = ClinicalWorldModel(state_dim=len(state_vars)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalized_states, state_masks, action_ids, current_raw_state = prepare_history(
        df=patient_df,
        current_index=current_index,
        state_vars=state_vars,
        train_mean=train_mean,
        train_std=train_std,
        tev_quantiles=tev_quantiles,
        ne_quantiles=ne_quantiles,
    )

    state_support_ok, state_support_reason = check_state_distribution(
        current_raw_state=current_raw_state,
        train_mean=train_mean,
        train_std=train_std,
        max_z=max_state_z,
    )

    states_tensor = torch.tensor(normalized_states[None, ...], dtype=torch.float32, device=device)
    masks_tensor = torch.tensor(state_masks[None, ...], dtype=torch.float32, device=device)
    base_actions = action_ids.copy()

    map_index = get_state_index(state_vars, "map")
    lactate_index = get_state_index(state_vars, "lactate")
    current_map = float(current_raw_state[map_index]) if map_index is not None else None
    current_lactate = float(current_raw_state[lactate_index]) if lactate_index is not None else None

    print("[5] 开始反事实模拟与可信度评估...")
    result_rows: List[Dict[str, Any]] = []

    for action in all_actions:
        action_id = int(action["action_id"])
        row = {
            "stay_id": stay_id,
            "time_index": float(current_time),
            "action_id": action_id,
            "fluid_bin": action["fluid_bin"],
            "fluid_level": action["fluid_level"],
            "vaso_bin": action["vaso_bin"],
            "vaso_level": action["vaso_level"],
            "guideline_status": action["merged_status"],
            "guideline_reason": action["reason"],
            "missing_evidence": action["missing_evidence"],
            "triggered_conditions": action["triggered_conditions"],
            "rule_assessments": action["rule_assessments"],
            "state_support_ok": state_support_ok,
            "state_support_reason": state_support_reason,
            "simulation_status": None,
            "rejection_reason": None,
            "predicted_map": None,
            "predicted_lactate": None,
            "map_change": None,
            "lactate_change": None,
            "mortality_probability": None,
            "vasopressor_requirement_probability": None,
            "mean_sigma_raw": None,
            "max_sigma_z": None,
            "action_support_count": None,
        }

        if action["merged_status"] == rules.STATUS_RULE_CONFLICT:
            row["simulation_status"] = "REJECT_RULE_CONFLICT"
            row["rejection_reason"] = "存在经规则引擎标记的明确规则冲突（待临床审核的保守判定）。"
            result_rows.append(row)
            continue

        action_ok, action_reason, action_count = check_action_support(action_id, checkpoint, min_action_count)
        row["action_support_count"] = action_count
        if not action_ok:
            row["simulation_status"] = "REJECT_LOW_ACTION_SUPPORT"
            row["rejection_reason"] = action_reason
            result_rows.append(row)
            continue

        if not state_support_ok:
            row["simulation_status"] = "REJECT_LOW_STATE_SUPPORT"
            row["rejection_reason"] = state_support_reason
            result_rows.append(row)
            continue

        candidate_actions = base_actions.copy()
        candidate_actions[-1] = action_id
        candidate_actions_tensor = torch.tensor(candidate_actions[None, ...], dtype=torch.long, device=device)

        with torch.no_grad():
            mu_norm, logvar, mortality_logit, vasopressor_logit = model(
                states=states_tensor, masks=masks_tensor, actions=candidate_actions_tensor
            )

        mu_norm_np = mu_norm[0].detach().cpu().numpy()
        logvar_np = logvar[0].detach().cpu().numpy()
        sigma_norm = np.exp(0.5 * np.clip(logvar_np, -30.0, 30.0))
        predicted_state = mu_norm_np * train_std + train_mean

        mortality_probability = float(torch.sigmoid(mortality_logit[0]).detach().cpu().item())
        vasopressor_probability = float(torch.sigmoid(vasopressor_logit[0]).detach().cpu().item())

        uncertainty_ok, uncertainty_reason, mean_sigma_raw = check_prediction_uncertainty(
            sigma_norm=sigma_norm, train_std=train_std, max_mean_sigma_raw=max_mean_sigma_raw, max_sigma_z=max_sigma_z
        )

        row["mean_sigma_raw"] = mean_sigma_raw
        row["max_sigma_z"] = float(np.max(sigma_norm))
        row["mortality_probability"] = mortality_probability
        row["vasopressor_requirement_probability"] = vasopressor_probability

        if not uncertainty_ok:
            row["simulation_status"] = "REJECT_HIGH_UNCERTAINTY"
            row["rejection_reason"] = uncertainty_reason
            result_rows.append(row)
            continue

        physiological_ok, physiological_reason = check_physiological_range(predicted_state, state_vars)
        if not physiological_ok:
            row["simulation_status"] = "REJECT_PHYSIOLOGICAL_RANGE"
            row["rejection_reason"] = physiological_reason
            result_rows.append(row)
            continue

        predicted_map = float(predicted_state[map_index]) if map_index is not None else None
        predicted_lactate = float(predicted_state[lactate_index]) if lactate_index is not None else None
        row["predicted_map"] = predicted_map
        row["predicted_lactate"] = predicted_lactate

        if predicted_map is not None and current_map is not None:
            row["map_change"] = predicted_map - current_map
        if predicted_lactate is not None and current_lactate is not None:
            row["lactate_change"] = predicted_lactate - current_lactate

        row["simulation_status"] = "SIMULATED_BUT_NOT_CLINICAL_RECOMMENDATION"
        result_rows.append(row)

    # ==========================================================
    # 移除复合打分后，排序机制：
    # 优先级 1：指南状态 (SUPPORTED > CONDITIONAL > INSUFFICIENT)
    # 优先级 2：预测死亡概率 (升序，越低越好)
    # 优先级 3：预测升压药需求概率 (升序，越低越好)
    # ==========================================================
    def get_status_priority(status: str) -> int:
        if status == rules.STATUS_SUPPORTED: return 3
        if status == rules.STATUS_CONDITIONAL: return 2
        if status == rules.STATUS_INSUFFICIENT: return 1
        return 0 

    accepted_rows = [
        row for row in result_rows
        if row["simulation_status"] == "SIMULATED_BUT_NOT_CLINICAL_RECOMMENDATION"
    ]

    # 多级排序
    accepted_rows.sort(
        key=lambda row: (
            get_status_priority(row["guideline_status"]),
            -row["mortality_probability"] if row["mortality_probability"] is not None else 1.0,
            -row["vasopressor_requirement_probability"] if row["vasopressor_requirement_probability"] is not None else 1.0,
        ),
        reverse=True
    )

    for rank, row in enumerate(accepted_rows, start=1):
        row["research_rank"] = rank

    print("\n" + "=" * 72)
    print(f"患者 {stay_id} 离线反事实候选结果")
    print("=" * 72)
    print("注意：以下结果仅基于指南状态与 CWM 结局预测展示，不构成临床治疗建议。")
    print(f"当前状态：MAP={current_map}, 乳酸={current_lactate}")
    print(f"未拒绝模拟动作：{len(accepted_rows)}")

    if not accepted_rows:
        print("没有动作同时通过指南、状态分布、不确定性和生理范围检查。")
    else:
        for index, row in enumerate(accepted_rows[:5], start=1): # 展示前 5 个
            print(f"[Top {index}] 补液 [{row['fluid_level']}] + 升压药 [{row['vaso_level']}] | 指南状态：{row['guideline_status']}")
            print(f"    CWM预测：MAP={row['predicted_map']:.2f}, 乳酸={row['predicted_lactate']:.2f}")
            print(f"    院内死亡预测概率：{row['mortality_probability']:.4f} | 未来升压药需求预测概率：{row['vasopressor_requirement_probability']:.4f}")
            print("-" * 72)

    rejected_counter = Counter(
        row["simulation_status"] for row in result_rows
        if row["simulation_status"] != "SIMULATED_BUT_NOT_CLINICAL_RECOMMENDATION"
    )
    print("拒绝/未模拟原因统计：" + json.dumps(dict(rejected_counter), ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"inference_{stay_id}.json"
    output_csv = output_dir / f"inference_{stay_id}.csv"

    with output_json.open("w", encoding="utf-8") as file:
        json.dump({
            "stay_id": stay_id,
            "time_index": float(current_time),
            "data_path": str(data_path),
            "checkpoint_path": str(checkpoint_path),
            "state_vars": state_vars,
            "guideline_status_counts": dict(status_counts),
            "insufficient_data_analysis": {
                "actions_with_g03_missing_fluid_responsiveness": insuff_g03_count,
                "actions_with_g05_missing_overload_evidence": insuff_g05_count
            },
            "result_interpretation": "仅用于离线研究，不构成临床治疗建议。",
            "results": result_rows,
        }, file, ensure_ascii=False, indent=2, default=str)

    csv_rows = []
    for row in result_rows:
        csv_row = row.copy()
        for key in ["missing_evidence", "triggered_conditions", "rule_assessments"]:
            csv_row[key] = json.dumps(csv_row[key], ensure_ascii=False)
        csv_rows.append(csv_row)

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nJSON 输出：{output_json}")
    print(f"CSV 输出：{output_csv}")


# ============================================================
# 10. 命令行入口 (阈值文献依据)
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="指南约束 + CWM 反事实可信度推理")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="MIMIC CSV 路径。")
    parser.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT), help="Clinical World Model checkpoint 路径。")
    parser.add_argument("--stay_id", type=str, default=None, help="需要推理的 stay_id。如果不填，默认选第一个患者。")
    parser.add_argument("--time_index", type=float, default=None, help="决策时间点。不指定时使用该患者最后一个时间点。")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")

    # 阈值参数的文献及统计学依据说明
    parser.add_argument(
        "--max_state_z", type=float, default=5.0,
        help="当前状态最大标准化偏离阈值。文献依据：在临床数据分布中，Z-score 超过 5 通常被认定为严重的 Out-of-Distribution (OOD) 异常值，超出模型安全泛化范围。"
    )
    parser.add_argument(
        "--max_mean_sigma_raw", type=float, default=5.0,
        help="预测状态原始尺度平均标准差阈值。文献依据：用于限制模型在所有特征上的平均不确定性，防止给出模棱两可的预测。"
    )
    parser.add_argument(
        "--max_sigma_z", type=float, default=2.5,
        help="预测标准差在标准化空间中的最大阈值。文献依据：贝叶斯深度学习文献通常将 > 2.0~2.5 的 sigma 视为高认知不确定性（Epistemic Uncertainty）警报线。"
    )

    parser.add_argument("--min_action_count", type=int, default=1, help="动作最小训练样本数。")
    args = parser.parse_args()

    run_pipeline(
        data_path=Path(args.data),
        checkpoint_path=Path(args.ckpt),
        stay_id=args.stay_id,
        time_index=args.time_index,
        output_dir=Path(args.output_dir),
        max_state_z=args.max_state_z,
        max_mean_sigma_raw=args.max_mean_sigma_raw,
        max_sigma_z=args.max_sigma_z,
        min_action_count=args.min_action_count,
    )

if __name__ == "__main__":
    main()