# -*- coding: utf-8 -*-
"""
指南约束模块（严格对标《SepsisAgent 指南约束模块：具体设计与实现方案》修订版）

修订说明（相对上一版的偏差修正）：
1. 状态枚举以方案文档第六节为准：
   RULE_CONFLICT / INSUFFICIENT_INFORMATION / CONDITIONAL / SUPPORTED_BY_GUIDELINE。
   文档第四步演示代码中的 UNKNOWN / CAUTION 仅为占位，不再使用。
2. 使 SUPPORTED_BY_GUIDELINE 可达：G02（补液后持续低血压 + 动作含升压药）
   与 G06（不稳定休克 + 联合治疗）在证据充分时按文档第六节第 4 级返回 SUPPORTED。
3. 使 RULE_CONFLICT 可达且保守：仅启用一条"待临床审核"的明确冲突
   （G05：已记录容量过负荷证据 + 高档/极高档补液），可通过常量开关关闭。
4. 所有文档未给出数值的阈值集中到 PROVISIONAL 常量区，标注"待临床审核"。
5. 为每条规则补充文档 YAML 示例要求的元数据：source / topic / required_fields。
6. 启用上下文中已提取但此前未使用的字段：
   weight_kg（G04/G05 以 mL/kg 形式呈现）、prior_vaso_level（G01 达标分支）。
7. 修复 infer_action_level_from_row 的整数歧义：仅当列名含 bin/level 时
   才允许把 0-4 的整数直接解释为档位。

关于 SUPPORTED_BY_GUIDELINE 可达性的说明：
- 规则级 SUPPORTED 已可达（G02、G06 在证据充分时返回）。
- 合并级 SUPPORTED 在实际数据中极难达到：按文档 YAML 示例，
  G01/G03 等"建议考虑（consider）"类规则的上限就是 CONDITIONAL，
  而合并规则要求"所有相关规则均满足"。这是文档原意的严格实现，
  即 SUPPORTED 只表示"当前规则库找到了明确适用依据"，
  不代表动作安全或最佳。如临床审核后认为某些"未触发"规则
  不应参与合并，可在 merge 时按 triggered_conditions 过滤。

重要原则（与文档一致）：
- 本模块不判断哪个动作治疗效果最好，只判断动作是否有临床依据、
  需要满足什么条件、是否存在风险或证据不足。
- 指南中的"建议考虑某治疗"不会被转换成"必须给予某剂量"。
- 三值逻辑：unknown 不等于 negative，缺失不当作阴性。
- 只使用当前决策时点及之前的数据。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. 固定数据路径
# ============================================================

DEFAULT_DATA_PATH = Path(
    r"C:\Users\恭喜发财\Desktop\data-1793668464627.csv"
)

DEFAULT_OUTPUT_JSON = Path("guideline_constraint_results.json")
DEFAULT_OUTPUT_CSV = Path("guideline_constraint_results.csv")


# ============================================================
# 2. 状态枚举（以方案文档第六节为准）
# ============================================================

UNKNOWN = "unknown"
POSITIVE = "positive"
NEGATIVE = "negative"

STATUS_RULE_CONFLICT = "RULE_CONFLICT"
STATUS_INSUFFICIENT = "INSUFFICIENT_INFORMATION"
STATUS_CONDITIONAL = "CONDITIONAL"
STATUS_SUPPORTED = "SUPPORTED_BY_GUIDELINE"

# 文档第六节规定的合并优先级（高 -> 低）：
# 1. 存在经过临床审核的明确规则冲突 -> RULE_CONFLICT
# 2. 无明确冲突，但关键必要信息缺失 -> INSUFFICIENT_INFORMATION
# 3. 无冲突，存在需进一步确认的条件 -> CONDITIONAL
# 4. 所有相关规则均满足，而且证据充分 -> SUPPORTED_BY_GUIDELINE
STATUS_PRIORITY = [
    STATUS_RULE_CONFLICT,
    STATUS_INSUFFICIENT,
    STATUS_CONDITIONAL,
    STATUS_SUPPORTED,
]


# ============================================================
# 3. 动作空间
#    文档定义：Fluid × Vasopressor = 5 × 5
# ============================================================

LEVEL_NAMES = {
    0: "None",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
}


@dataclass(frozen=True)
class TreatmentAction:
    action_id: int
    fluid_bin: int
    vaso_bin: int

    @property
    def fluid_level(self) -> str:
        return LEVEL_NAMES[self.fluid_bin]

    @property
    def vaso_level(self) -> str:
        return LEVEL_NAMES[self.vaso_bin]


@dataclass
class GuidelinePatientContext:
    patient_id: Any = None
    time_index: Any = None

    age: Optional[float] = None
    map_current: Optional[float] = None
    map_trend: Optional[float] = None
    map_history: Optional[List[float]] = None

    lactate_current: Optional[float] = None
    lactate_trend: Optional[float] = None
    lactate_history: Optional[List[float]] = None

    fluid_last_4h_ml: Optional[float] = None
    fluid_cumulative_ml: Optional[float] = None
    fluid_cumulative_source: str = "unknown"

    weight_kg: Optional[float] = None

    vaso_current_nee: Optional[float] = None
    vaso_trend: Optional[str] = None

    fluid_responsiveness: str = UNKNOWN
    fluid_overload_evidence: str = UNKNOWN
    shock_instability: str = UNKNOWN

    prior_fluid_level: Optional[int] = None
    prior_vaso_level: Optional[int] = None

    available_fields: Optional[List[str]] = None
    missing_fields: Optional[List[str]] = None


# ============================================================
# 3.5 临时阈值常量区（provisional，全部待临床审核）
#
# 文档只规定了规则的临床逻辑方向，未给出以下具体数值。
# 按文档第二节的要求："剂量档位如何映射、哪些情况构成明确冲突，
# 还需要临床审核。" 审核通过后应移入 YAML 规则库并记录 source 与版本。
# ============================================================

PROVISIONAL = {
    # G04：初始复苏与后续复苏的累计补液分界（mL）
    "g04_early_cumulative_ml": 1000.0,
    "g04_late_cumulative_ml": 3000.0,
    # 休克不稳定性保守推断（仅在无记录字段时作为兜底，证据不足一律 unknown）
    "shock_map_severe": 55.0,
    "shock_map_target": 65.0,
    "shock_lactate": 2.0,
    "shock_map_trend_drop": -5.0,
    # 读取既往动作记录时使用的连续剂量分档阈值（待临床审核；
    # 注意：这与训练脚本按训练集五分位数生成的 action 分档不是同一套刻度，
    # 仅用于 prior_fluid_level / prior_vaso_level 的近似展示）
    "prior_fluid_thresholds": [0.0, 250.0, 500.0, 1000.0],
    "prior_vaso_thresholds": [0.0, 0.02, 0.08, 0.2],
}

# G05 明确规则冲突开关。
# 文档第六节要求 RULE_CONFLICT 必须是"经过临床审核的明确规则冲突"。
# 第一版仅启用一条保守冲突，且集中定义于此，便于临床审核时逐条确认或关闭：
ENABLE_RULE_CONFLICT = True
RULE_CONFLICT_DEFINITIONS = {
    "G05": (
        "已记录容量过负荷证据（positive）时仍给予高档/极高档补液"
        "（fluid_bin >= 3）——与指南'结合心肺情况及液体负荷再评估'直接矛盾。"
        "待临床审核。"
    ),
}


# ============================================================
# 3.6 规则元数据（对应文档 YAML 示例的 source/topic/required_fields）
# ============================================================

RULE_METADATA: Dict[str, Dict[str, Any]] = {
    "G01": {
        "source": "SSC_2026",
        "topic": "vasopressor",
        "required_fields": ["age", "map_current"],
    },
    "G02": {
        "source": "SSC_2026",
        "topic": "vasopressor/fluid",
        "required_fields": ["map_current", "fluid_cumulative_ml"],
    },
    "G03": {
        "source": "SSC_2026",
        "topic": "fluid",
        "required_fields": ["fluid_responsiveness"],
    },
    "G04": {
        "source": "SSC_2026",
        "topic": "fluid",
        "required_fields": ["fluid_cumulative_ml", "resuscitation_phase"],
    },
    "G05": {
        "source": "SSC_2026",
        "topic": "fluid",
        "required_fields": ["fluid_overload_evidence"],
    },
    "G06": {
        "source": "SSC_2026",
        "topic": "fluid+vasopressor",
        "required_fields": ["shock_instability"],
    },
}


# ============================================================
# 4. 字段识别
# ============================================================

def normalize_column_name(value: Any) -> str:
    """
    将列名标准化，用于自动匹配不同命名方式。
    例如：
    meanbp、mean_bp、MAP、map_current
    """
    text = str(value).strip().lower()
    text = text.replace("μ", "u")
    text = text.replace("µ", "u")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


COLUMN_ALIASES: Dict[str, List[str]] = {
    "patient_id": [
        "stay_id",
        "icustay_id",
        "subject_id",
        "patient_id",
        "hadm_id",
        "episode_id",
    ],
    "time_index": [
        "hour_from_onset",
        "bin_start",
        "timestep",
        "time_step",
        "step",
        "time",
        "hour",
        "icu_hour",
        "hours_from_icu",
        "hours_since_icu",
        "charttime",
        "intime",
    ],
    "age": [
        "age",
        "age_years",
        "patient_age",
    ],
    "map": [
        "map",
        "meanbp",
        "mean_bp",
        "map_current",
        "mean_arterial_pressure",
        "mean_arterial_pressure_mmhg",
    ],
    "lactate": [
        "lactate",
        "lactate_current",
        "lactate_mmol_l",
        "lactate_mmol",
    ],
    "weight": [
        "weight",
        "weight_kg",
        "admission_weight",
        "weight_kg_current",
    ],
    "fluid_last_4h": [
        "fluid_last_4h_ml",
        "fluid_4h_ml",
        "iv_fluid_4h_ml",
        "tev_ml",
        "tev",
        "total_effective_volume",
        "total_effective_volume_ml",
        "fluid_volume",
        "fluid_ml",
        "iv_fluid",
    ],
    "fluid_cumulative": [
        "fluid_cumulative_ml",
        "icu_recorded_cumulative_fluid_ml",
        "cumulative_fluid",
        "cumulative_fluid_ml",
        "cumulative_tev",
        "cumulative_tev_ml",
    ],
    "vaso_nee": [
        "vaso_current_nee",
        "ne_eq",
        "nee",
        "norepinephrine_equivalent",
        "norepinephrine_equivalent_mcg_kg_min",
        "vasopressor",
        "vasopressor_rate",
        "vaso_rate",
    ],
    "fluid_responsiveness": [
        "fluid_responsiveness",
        "fluid_response",
        "fluid_responsive",
        "dynamic_fluid_response",
    ],
    "fluid_overload": [
        "fluid_overload_evidence",
        "fluid_overload",
        "volume_overload",
        "overload_evidence",
    ],
    "shock_instability": [
        "shock_instability",
        "unstable_shock",
        "hemodynamic_instability",
        "shock_status",
    ],
    "fluid_action": [
        "fluid_bin",
        "iv_fluid_level",
        "fluid_level",
        "action_fluid",
        "a_fluid",
    ],
    "vaso_action": [
        "vaso_bin",
        "vasopressor_level",
        "vaso_level",
        "action_vaso",
        "a_vaso",
    ],
}


def build_normalized_column_map(columns: Iterable[Any]) -> Dict[str, str]:
    return {
        normalize_column_name(column): str(column)
        for column in columns
    }


def find_column(
    normalized_columns: Dict[str, str],
    logical_name: str,
) -> Optional[str]:
    aliases = COLUMN_ALIASES.get(logical_name, [])

    normalized_aliases = [
        normalize_column_name(alias)
        for alias in aliases
    ]

    # 第一优先级：完全匹配
    for alias in normalized_aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    # 第二优先级：仅在字段以下划线分隔的情况下匹配
    for column_normalized, original_column in normalized_columns.items():
        column_tokens = set(column_normalized.split("_"))

        for alias in normalized_aliases:
            alias_tokens = set(alias.split("_"))

            if alias_tokens.issubset(column_tokens):
                return original_column

    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    normalized_columns = build_normalized_column_map(df.columns)

    detected = {
        logical_name: find_column(normalized_columns, logical_name)
        for logical_name in COLUMN_ALIASES
    }

    return detected


# ============================================================
# 5. 数据读取
# ============================================================

def read_csv_strict(data_path: Path) -> pd.DataFrame:
    """
    只读取用户指定路径，不加载任何内置数据。
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{data_path}"
        )

    if not data_path.is_file():
        raise ValueError(
            f"指定路径不是文件：{data_path}"
        )

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]

    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                data_path,
                encoding=encoding,
                low_memory=False,
            )
            if df.empty:
                raise ValueError(
                    f"CSV 文件为空：{data_path}"
                )
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"无法读取 CSV 文件：{data_path}\n"
        f"最后一次错误：{last_error}"
    )


# ============================================================
# 6. 数值和状态转换
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    """
    将数值、字符串数值安全转换为 float。

    注意：
    - 缺失值返回 None；
    - 无穷值返回 None；
    - 字符串中的单位和逗号会被移除；
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        text = text.replace(",", "")

        # 保留数字、正负号、小数点和科学计数法
        text = re.sub(r"[^0-9eE+\-.]", "", text)

        if not text:
            return None

        value_to_convert = text
    else:
        value_to_convert = value

    try:
        result = float(value_to_convert)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def normalize_tri_state(value: Any) -> str:
    """
    不把缺失值当作阴性。
    """
    if value is None or pd.isna(value):
        return UNKNOWN

    text = str(value).strip().lower()

    if text in {
        "positive",
        "pos",
        "yes",
        "true",
        "responsive",
        "present",
        "high",
        "overload",
        "unstable",
        "increasing",
    }:
        return POSITIVE

    if text in {
        "negative",
        "neg",
        "no",
        "false",
        "non-responsive",
        "nonresponsive",
        "absent",
        "none",
        "stable",
        "decreasing",
    }:
        return NEGATIVE

    if text in {
        "unknown",
        "unk",
        "missing",
        "na",
        "n/a",
        "not_available",
        "not tested",
        "not_tested",
        "",
    }:
        return UNKNOWN

    return UNKNOWN


def normalize_trend(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in {
        "increasing",
        "increase",
        "up",
        "rising",
        "升高",
        "上升",
    }:
        return "increasing"

    if text in {
        "decreasing",
        "decrease",
        "down",
        "falling",
        "下降",
        "降低",
    }:
        return "decreasing"

    if text in {
        "stable",
        "unchanged",
        "平稳",
        "稳定",
    }:
        return "stable"

    return text


def clean_numeric_history(
    values: Sequence[Any],
) -> List[float]:
    result = []

    for value in values:
        numeric = safe_float(value)
        if numeric is not None:
            result.append(numeric)

    return result


def infer_bin_from_value(
    value: Any,
    thresholds: Sequence[float],
) -> Optional[int]:
    """
    将动作历史中的连续剂量映射到 0-4 档。
    注意：这只是用于读取已有动作记录，不是将指南建议转换成剂量。
    """
    numeric = safe_float(value)

    if numeric is None:
        return None

    for index, threshold in enumerate(thresholds):
        if numeric <= threshold:
            return index

    return len(thresholds)


# ============================================================
# 7. 数据排序和患者轨迹处理
# ============================================================

def sort_trajectory(
    trajectory: pd.DataFrame,
    time_column: Optional[str],
) -> pd.DataFrame:
    if time_column is None:
        return trajectory.reset_index(drop=True)

    numeric_time = pd.to_numeric(
        trajectory[time_column],
        errors="coerce",
    )

    if numeric_time.notna().any():
        return (
            trajectory.assign(_sort_time=numeric_time)
            .sort_values("_sort_time")
            .drop(columns=["_sort_time"])
            .reset_index(drop=True)
        )

    return trajectory.sort_values(
        by=time_column,
        kind="stable",
    ).reset_index(drop=True)


def get_patient_groups(
    df: pd.DataFrame,
    detected: Dict[str, Optional[str]],
) -> List[Tuple[Any, pd.DataFrame]]:
    patient_column = detected.get("patient_id")
    time_column = detected.get("time_index")

    if patient_column is None:
        return [(None, sort_trajectory(df, time_column))]

    groups = []

    for patient_id, group in df.groupby(
        patient_column,
        dropna=False,
        sort=False,
    ):
        groups.append(
            (
                patient_id,
                sort_trajectory(group, time_column),
            )
        )

    return groups


# ============================================================
# 8. 从历史轨迹构建 GuidelinePatientContext
# ============================================================

def calculate_trend(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None

    return float(values[-1] - values[-2])


def infer_shock_instability(
    map_current: Optional[float],
    map_trend: Optional[float],
    lactate_current: Optional[float],
    vaso_current: Optional[float],
) -> str:
    """
    只根据现有可解释变量进行保守判断。
    如果证据不充分，返回 unknown。
    阈值来自 PROVISIONAL 常量区（待临床审核）。
    """
    if map_current is None or lactate_current is None:
        return UNKNOWN

    if (
        map_current < PROVISIONAL["shock_map_severe"]
        and lactate_current > PROVISIONAL["shock_lactate"]
    ):
        return POSITIVE

    if (
        map_current < PROVISIONAL["shock_map_target"]
        and lactate_current > PROVISIONAL["shock_lactate"]
        and vaso_current is not None
        and vaso_current > 0
    ):
        return POSITIVE

    if (
        map_current >= PROVISIONAL["shock_map_target"]
        and lactate_current <= PROVISIONAL["shock_lactate"]
    ):
        return NEGATIVE

    if (
        map_trend is not None
        and map_trend < PROVISIONAL["shock_map_trend_drop"]
    ):
        return POSITIVE

    return UNKNOWN


def infer_action_level_from_row(
    row: pd.Series,
    column: Optional[str],
    thresholds: Sequence[float],
) -> Optional[int]:
    """
    从当前行读取既往动作档位。

    修复说明（整数歧义）：仅当列名本身包含 bin/level 时，
    才允许把 0-4 的整数直接解释为档位；否则一律按连续剂量
    用阈值分档，避免把 4 mL 这类剂量误判为 4 档。
    """
    if column is None:
        return None

    value = row.get(column)

    numeric = safe_float(value)
    if numeric is None:
        return None

    normalized_column = normalize_column_name(column)
    column_is_level_like = (
        "bin" in normalized_column.split("_")
        or "level" in normalized_column.split("_")
    )

    if column_is_level_like:
        integer_like = abs(numeric - round(numeric)) < 1e-8
        if integer_like and 0 <= int(round(numeric)) <= 4:
            return int(round(numeric))

    return infer_bin_from_value(
        numeric,
        thresholds,
    )


def build_guideline_context(
    trajectory: pd.DataFrame,
    detected: Dict[str, Optional[str]],
    decision_index: Optional[int] = None,
) -> GuidelinePatientContext:
    """
    只使用当前决策时点及其之前的历史数据。

    如果 decision_index=None，则使用该患者轨迹最后一个时点。
    """
    if trajectory.empty:
        raise ValueError("患者轨迹为空，无法构建临床状态。")

    if decision_index is None:
        decision_index = len(trajectory) - 1

    if decision_index < 0 or decision_index >= len(trajectory):
        raise IndexError(
            f"decision_index 超出范围：{decision_index}"
        )

    history = trajectory.iloc[: decision_index + 1]
    current = history.iloc[-1]

    map_column = detected.get("map")
    lactate_column = detected.get("lactate")
    age_column = detected.get("age")
    weight_column = detected.get("weight")
    fluid_4h_column = detected.get("fluid_last_4h")
    fluid_cumulative_column = detected.get("fluid_cumulative")
    vaso_column = detected.get("vaso_nee")
    fluid_response_column = detected.get("fluid_responsiveness")
    overload_column = detected.get("fluid_overload")
    instability_column = detected.get("shock_instability")

    map_history = []
    if map_column is not None:
        map_history = clean_numeric_history(
            history[map_column].tolist()
        )

    lactate_history = []
    if lactate_column is not None:
        lactate_history = clean_numeric_history(
            history[lactate_column].tolist()
        )

    map_current = (
        map_history[-1]
        if map_history
        else None
    )

    lactate_current = (
        lactate_history[-1]
        if lactate_history
        else None
    )

    age = (
        safe_float(current.get(age_column))
        if age_column is not None
        else None
    )

    weight = (
        safe_float(current.get(weight_column))
        if weight_column is not None
        else None
    )

    fluid_last_4h = (
        safe_float(current.get(fluid_4h_column))
        if fluid_4h_column is not None
        else None
    )

    fluid_cumulative = None
    fluid_cumulative_source = "unknown"

    if fluid_cumulative_column is not None:
        fluid_cumulative = safe_float(
            current.get(fluid_cumulative_column)
        )
        if fluid_cumulative is not None:
            fluid_cumulative_source = (
                "icu_recorded_cumulative_fluid_ml"
            )

    if fluid_cumulative is None and fluid_4h_column is not None:
        numeric_fluid_history = [
            safe_float(value)
            for value in history[fluid_4h_column].tolist()
        ]
        numeric_fluid_history = [
            value
            for value in numeric_fluid_history
            if value is not None
        ]

        if numeric_fluid_history:
            fluid_cumulative = float(
                sum(numeric_fluid_history)
            )
            fluid_cumulative_source = (
                "sum_of_recorded_4h_fluid_field"
            )

    vaso_current = None

    if vaso_column is not None:
        vaso_current = safe_float(
            current.get(vaso_column)
        )

    vaso_history: List[float] = []

    if vaso_column is not None:
        vaso_history = clean_numeric_history(
            history[vaso_column].tolist()
        )

    vaso_trend: Optional[str] = None

    if len(vaso_history) >= 2:
        if vaso_history[-1] > vaso_history[-2]:
            vaso_trend = "increasing"
        elif vaso_history[-1] < vaso_history[-2]:
            vaso_trend = "decreasing"
        else:
            vaso_trend = "stable"

    fluid_response = (
        normalize_tri_state(
            current.get(fluid_response_column)
        )
        if fluid_response_column is not None
        else UNKNOWN
    )

    overload_evidence = (
        normalize_tri_state(
            current.get(overload_column)
        )
        if overload_column is not None
        else UNKNOWN
    )

    shock_instability = (
        normalize_tri_state(
            current.get(instability_column)
        )
        if instability_column is not None
        else infer_shock_instability(
            map_current=map_current,
            map_trend=calculate_trend(map_history),
            lactate_current=lactate_current,
            vaso_current=vaso_current,
        )
    )

    fluid_action_column = detected.get("fluid_action")
    vaso_action_column = detected.get("vaso_action")

    prior_fluid_level = infer_action_level_from_row(
        current,
        fluid_action_column,
        thresholds=PROVISIONAL["prior_fluid_thresholds"],
    )

    prior_vaso_level = infer_action_level_from_row(
        current,
        vaso_action_column,
        thresholds=PROVISIONAL["prior_vaso_thresholds"],
    )

    patient_column = detected.get("patient_id")
    time_column = detected.get("time_index")

    patient_id = (
        current.get(patient_column)
        if patient_column is not None
        else None
    )

    time_index = (
        current.get(time_column)
        if time_column is not None
        else decision_index
    )

    relevant_logical_fields = [
        "age",
        "map",
        "lactate",
        "fluid_last_4h",
        "fluid_cumulative",
        "weight",
        "vaso_nee",
        "fluid_responsiveness",
        "fluid_overload",
        "shock_instability",
    ]

    available_fields = [
        field
        for field in relevant_logical_fields
        if detected.get(field) is not None
    ]

    missing_fields = [
        field
        for field in relevant_logical_fields
        if detected.get(field) is None
    ]

    return GuidelinePatientContext(
        patient_id=patient_id,
        time_index=time_index,
        age=age,
        map_current=map_current,
        map_trend=calculate_trend(map_history),
        map_history=map_history,
        lactate_current=lactate_current,
        lactate_trend=calculate_trend(lactate_history),
        lactate_history=lactate_history,
        fluid_last_4h_ml=fluid_last_4h,
        fluid_cumulative_ml=fluid_cumulative,
        fluid_cumulative_source=fluid_cumulative_source,
        weight_kg=weight,
        vaso_current_nee=vaso_current,
        vaso_trend=vaso_trend,
        fluid_responsiveness=fluid_response,
        fluid_overload_evidence=overload_evidence,
        shock_instability=shock_instability,
        prior_fluid_level=prior_fluid_level,
        prior_vaso_level=prior_vaso_level,
        available_fields=available_fields,
        missing_fields=missing_fields,
    )


def fluid_cumulative_per_kg_note(ctx: GuidelinePatientContext) -> str:
    """
    当体重可得时，以 mL/kg 形式呈现累计补液，供 G04/G05 结合体重评估。
    不引入文档之外的 mL/kg 硬性截断，仅作呈现。
    """
    if (
        ctx.fluid_cumulative_ml is None
        or ctx.weight_kg is None
        or ctx.weight_kg <= 0
    ):
        return ""

    per_kg = ctx.fluid_cumulative_ml / ctx.weight_kg
    return f"按体重折算约 {per_kg:.1f} mL/kg；"


# ============================================================
# 9. 规则结果
# ============================================================

def rule_result(
    rule_id: str,
    name: str,
    status: str,
    reason: str,
    triggered_conditions: Optional[List[str]] = None,
    missing_evidence: Optional[List[str]] = None,
    clinical_scope: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = RULE_METADATA.get(rule_id, {})

    return {
        "rule_id": rule_id,
        "name": name,
        # 文档 YAML 示例要求的元数据字段
        "source": metadata.get("source"),
        "topic": metadata.get("topic"),
        "required_fields": metadata.get("required_fields", []),
        "status": status,
        "reason": reason,
        "triggered_conditions": triggered_conditions or [],
        "missing_evidence": missing_evidence or [],
        "clinical_scope": clinical_scope
        or (
            "该规则只评估当前动作是否具有临床判断背景，"
            "不代表该动作是最佳治疗，也不代表动作一定安全。"
        ),
    }


# ============================================================
# 10. G01：MAP 目标
# ============================================================

def assess_g01_map_target(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    if ctx.map_current is None:
        return rule_result(
            rule_id="G01",
            name="MAP target",
            status=STATUS_INSUFFICIENT,
            reason="当前 MAP 缺失，无法判断是否低于初始目标。",
            missing_evidence=["map_current"],
        )

    if ctx.age is None:
        return rule_result(
            rule_id="G01",
            name="MAP target",
            status=STATUS_INSUFFICIENT,
            reason=(
                "年龄缺失，无法区分普通成人目标和文档中"
                "老年患者 60-65 mmHg 的目标范围。"
            ),
            missing_evidence=["age"],
        )

    if ctx.age >= 65:
        target_low = 60.0
        target_high = 65.0
        target_description = "老年患者初始 MAP 参考范围 60-65 mmHg"
    else:
        target_low = 65.0
        target_high = 65.0
        target_description = "普通成人初始 MAP 参考目标约 65 mmHg"

    if ctx.map_current < target_low:
        if action.vaso_bin == 0:
            return rule_result(
                rule_id="G01",
                name="MAP target",
                status=STATUS_CONDITIONAL,
                reason=(
                    f"当前 MAP={ctx.map_current:.2f} mmHg，"
                    f"低于{target_description}；当前动作未包含升压药，"
                    "应进一步评估是否需要升压药支持。"
                ),
                triggered_conditions=[
                    "map_below_initial_target",
                    "action_has_no_vasopressor",
                ],
            )

        return rule_result(
            rule_id="G01",
            name="MAP target",
            status=STATUS_CONDITIONAL,
            reason=(
                f"当前 MAP={ctx.map_current:.2f} mmHg，"
                f"低于{target_description}；该动作包含升压药，"
                "具有需要考虑升压支持的临床背景。"
            ),
            triggered_conditions=[
                "map_below_initial_target",
                "action_has_vasopressor",
            ],
        )

    if ctx.map_current > target_high:
        if action.vaso_bin >= 3:
            return rule_result(
                rule_id="G01",
                name="MAP target",
                status=STATUS_CONDITIONAL,
                reason=(
                    f"当前 MAP={ctx.map_current:.2f} mmHg，"
                    "已经高于初始目标；该动作仍包含高档升压药，"
                    "需要重新评估持续升压的依据。"
                ),
                triggered_conditions=[
                    "map_at_or_above_target",
                    "action_has_high_vasopressor",
                ],
            )

        # 文档明确提醒：MAP 达标不能推断"所有升压药都不必要"，
        # 有些患者可能正在通过现有升压药维持目标血压。
        # 此处启用 prior_vaso_level 进行该判断。
        if (
            ctx.prior_vaso_level is not None
            and ctx.prior_vaso_level > 0
            and action.vaso_bin > 0
        ):
            return rule_result(
                rule_id="G01",
                name="MAP target",
                status=STATUS_CONDITIONAL,
                reason=(
                    f"当前 MAP={ctx.map_current:.2f} mmHg，"
                    "已经高于初始目标；但既往记录提示患者可能正在"
                    "通过现有升压药维持目标血压，不能仅因 MAP 达标"
                    "推断升压药不必要，需要结合灌注情况复评剂量。"
                ),
                triggered_conditions=[
                    "map_at_or_above_target",
                    "prior_vasopressor_recorded",
                    "possible_vasopressor_maintained_map",
                ],
            )

        return rule_result(
            rule_id="G01",
            name="MAP target",
            status=STATUS_CONDITIONAL,
            reason=(
                f"当前 MAP={ctx.map_current:.2f} mmHg，"
                "未发现仅由 MAP 目标规则导致的明确冲突。"
            ),
            triggered_conditions=[
                "map_at_or_above_target",
            ],
        )

    return rule_result(
        rule_id="G01",
        name="MAP target",
        status=STATUS_CONDITIONAL,
        reason=(
            f"当前 MAP={ctx.map_current:.2f} mmHg，"
            f"位于{target_description}附近。"
        ),
        triggered_conditions=[
            "map_near_initial_target",
        ],
    )


# ============================================================
# 11. G02：持续低血压与既往复苏
# ============================================================

def assess_g02_persistent_hypotension(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    missing = []

    if ctx.map_current is None:
        missing.append("map_current")

    if ctx.fluid_cumulative_ml is None:
        missing.append("fluid_cumulative_ml")

    if missing:
        return rule_result(
            rule_id="G02",
            name="persistent hypotension after fluid",
            status=STATUS_INSUFFICIENT,
            reason=(
                "无法同时判断当前低血压和既往记录到的补液情况。"
            ),
            missing_evidence=missing,
        )

    if ctx.map_current < 65:
        if ctx.fluid_cumulative_ml > 0:
            if action.vaso_bin == 0:
                return rule_result(
                    rule_id="G02",
                    name="persistent hypotension after fluid",
                    status=STATUS_CONDITIONAL,
                    reason=(
                        f"当前 MAP={ctx.map_current:.2f} mmHg，"
                        f"记录到既往补液累计约 "
                        f"{ctx.fluid_cumulative_ml:.2f} mL，"
                        "低血压仍存在；该动作没有升压药，"
                        "需要进一步评估持续低血压风险。"
                    ),
                    triggered_conditions=[
                        "map_below_65",
                        "prior_fluid_recorded",
                        "persistent_hypotension_possible",
                    ],
                )

            # 修订：证据充分（低血压 + 既往补液已记录 + 动作含升压药）
            # 时，按文档第六节第 4 级返回 SUPPORTED_BY_GUIDELINE。
            # 含义仅为"当前规则库找到了适用依据"，不代表最佳治疗或安全。
            return rule_result(
                rule_id="G02",
                name="persistent hypotension after fluid",
                status=STATUS_SUPPORTED,
                reason=(
                    f"当前 MAP={ctx.map_current:.2f} mmHg，"
                    f"记录到既往补液累计约 "
                    f"{ctx.fluid_cumulative_ml:.2f} mL，"
                    "低血压仍存在；动作包含升压药支持，"
                    "符合补液后持续低血压需升压支持的指南背景，"
                    "当前规则库找到了适用依据。"
                ),
                triggered_conditions=[
                    "map_below_65",
                    "prior_fluid_recorded",
                    "persistent_hypotension_possible",
                    "action_has_vasopressor",
                ],
            )

        return rule_result(
            rule_id="G02",
            name="persistent hypotension after fluid",
            status=STATUS_INSUFFICIENT,
            reason=(
                "当前存在低血压，但既往补液记录显示为零，"
                "不能直接判断是否属于补液后持续低血压。"
            ),
            missing_evidence=[
                "adequate_prior_resuscitation",
            ],
        )

    return rule_result(
        rule_id="G02",
        name="persistent hypotension after fluid",
        status=STATUS_CONDITIONAL,
        reason=(
            "当前 MAP 未低于 65 mmHg，未发现持续低血压规则"
            "导致的明确冲突。"
        ),
        triggered_conditions=[
            "map_not_below_65",
        ],
    )


# ============================================================
# 12. G03：液体反应性
#     严格对应文档 YAML 示例：
#     positive -> CONDITIONAL；negative -> CONDITIONAL（谨慎）；
#     unknown -> INSUFFICIENT_INFORMATION。
# ============================================================

def assess_g03_fluid_responsiveness(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    if action.fluid_bin == 0:
        return rule_result(
            rule_id="G03",
            name="fluid responsiveness",
            status=STATUS_CONDITIONAL,
            reason=(
                "当前动作不追加补液，因此液体反应性规则"
                "不会对该动作构成追加补液限制。"
            ),
            triggered_conditions=[
                "no_additional_fluid_action",
            ],
        )

    if ctx.fluid_responsiveness == POSITIVE:
        return rule_result(
            rule_id="G03",
            name="fluid responsiveness",
            status=STATUS_CONDITIONAL,
            reason=(
                "已记录液体反应性为阳性；追加补液具有"
                "进一步评估后可以考虑的临床依据。"
            ),
            triggered_conditions=[
                "fluid_responsiveness_positive",
                "additional_fluid_action",
            ],
        )

    if ctx.fluid_responsiveness == NEGATIVE:
        if action.fluid_bin >= 3:
            return rule_result(
                rule_id="G03",
                name="fluid responsiveness",
                status=STATUS_CONDITIONAL,
                reason=(
                    "已记录液体反应性为阴性；当前动作仍为"
                    "高档或极高档补液，需要谨慎评估。"
                ),
                triggered_conditions=[
                    "fluid_responsiveness_negative",
                    "high_additional_fluid_action",
                ],
            )

        return rule_result(
            rule_id="G03",
            name="fluid responsiveness",
            status=STATUS_CONDITIONAL,
            reason=(
                "液体反应性为阴性；当前追加补液需要"
                "结合其他灌注证据和容量风险进行判断。"
            ),
            triggered_conditions=[
                "fluid_responsiveness_negative",
                "additional_fluid_action",
            ],
        )

    return rule_result(
        rule_id="G03",
        name="fluid responsiveness",
        status=STATUS_INSUFFICIENT,
        reason=(
            "没有可靠的液体反应性证据。unknown 不等于阴性，"
            "因此不能据此支持追加补液，也不能据此认定追加补液错误。"
        ),
        triggered_conditions=[
            "additional_fluid_action",
            "fluid_responsiveness_unknown",
        ],
        missing_evidence=[
            "fluid_responsiveness",
            "PLR_or_SV_or_SVV_or_PPV",
        ],
    )


# ============================================================
# 13. G04：累计补液及复苏阶段
#     阈值来自 PROVISIONAL 常量区（待临床审核）；
#     体重可得时按文档"结合容量情况"以 mL/kg 呈现。
# ============================================================

def assess_g04_cumulative_fluid(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    if action.fluid_bin == 0:
        return rule_result(
            rule_id="G04",
            name="cumulative fluid and resuscitation phase",
            status=STATUS_CONDITIONAL,
            reason=(
                "当前动作不追加补液。"
            ),
            triggered_conditions=[
                "no_additional_fluid_action",
            ],
        )

    if ctx.fluid_cumulative_ml is None:
        return rule_result(
            rule_id="G04",
            name="cumulative fluid and resuscitation phase",
            status=STATUS_INSUFFICIENT,
            reason=(
                "缺少累计补液量，无法区分初始复苏和后续追加补液阶段。"
            ),
            missing_evidence=[
                "fluid_cumulative_ml",
                "resuscitation_phase",
            ],
        )

    per_kg_note = fluid_cumulative_per_kg_note(ctx)

    if ctx.fluid_cumulative_ml < PROVISIONAL["g04_early_cumulative_ml"]:
        return rule_result(
            rule_id="G04",
            name="cumulative fluid and resuscitation phase",
            status=STATUS_CONDITIONAL,
            reason=(
                f"当前记录累计补液约 "
                f"{ctx.fluid_cumulative_ml:.2f} mL（{per_kg_note}"
                "来源："
                f"{ctx.fluid_cumulative_source}），"
                "仍可能处于早期复苏阶段；追加补液需要结合"
                "低灌注表现和液体反应性判断。"
            ),
            triggered_conditions=[
                "early_or_limited_recorded_resuscitation",
                "additional_fluid_action",
            ],
        )

    if ctx.fluid_cumulative_ml >= PROVISIONAL["g04_late_cumulative_ml"]:
        return rule_result(
            rule_id="G04",
            name="cumulative fluid and resuscitation phase",
            status=STATUS_CONDITIONAL,
            reason=(
                f"当前记录累计补液约 "
                f"{ctx.fluid_cumulative_ml:.2f} mL（{per_kg_note}"
                "来源："
                f"{ctx.fluid_cumulative_source}），"
                "可能已经进入后续复苏阶段；继续追加高档补液"
                "需要更强的临床依据。"
            ),
            triggered_conditions=[
                "substantial_recorded_cumulative_fluid",
                "later_resuscitation_phase_possible",
                "additional_fluid_action",
            ],
        )

    return rule_result(
        rule_id="G04",
        name="cumulative fluid and resuscitation phase",
        status=STATUS_CONDITIONAL,
        reason=(
            f"当前记录累计补液约 "
            f"{ctx.fluid_cumulative_ml:.2f} mL（{per_kg_note}"
            "来源："
            f"{ctx.fluid_cumulative_source}）；"
            "补液阶段可以部分判断，但仍需要结合"
            "动态灌注和容量风险。"
        ),
        triggered_conditions=[
            "recorded_cumulative_fluid",
            "additional_fluid_action",
        ],
    )


# ============================================================
# 14. G05：容量过负荷风险
#     修订：已记录容量过负荷证据 + 高档/极高档补液，
#     按文档第六节第 1 级返回 RULE_CONFLICT（待临床审核，
#     见 RULE_CONFLICT_DEFINITIONS，可通过开关关闭）。
# ============================================================

def assess_g05_fluid_overload(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    if action.fluid_bin < 3:
        return rule_result(
            rule_id="G05",
            name="fluid overload risk",
            status=STATUS_CONDITIONAL,
            reason=(
                "当前动作不是高档补液，不触发高档补液的"
                "容量过负荷重点审查。"
            ),
            triggered_conditions=[
                "not_high_fluid_action",
            ],
        )

    per_kg_note = fluid_cumulative_per_kg_note(ctx)

    if ctx.fluid_overload_evidence == POSITIVE:
        if ENABLE_RULE_CONFLICT and "G05" in RULE_CONFLICT_DEFINITIONS:
            return rule_result(
                rule_id="G05",
                name="fluid overload risk",
                status=STATUS_RULE_CONFLICT,
                reason=(
                    "已记录容量过负荷证据，当前动作仍为高档/极高档补液"
                    f"（fluid_bin={action.fluid_bin}）；"
                    f"{RULE_CONFLICT_DEFINITIONS['G05']}"
                    f"{per_kg_note}"
                    "该判定为保守规则冲突，仍需临床审核确认。"
                ),
                triggered_conditions=[
                    "high_fluid_action",
                    "fluid_overload_evidence_present",
                    "audited_rule_conflict_candidate",
                ],
            )

        return rule_result(
            rule_id="G05",
            name="fluid overload risk",
            status=STATUS_CONDITIONAL,
            reason=(
                "已记录容量过负荷证据；当前高档补液动作"
                "需要谨慎，并应进一步复核心肺和容量状态。"
            ),
            triggered_conditions=[
                "high_fluid_action",
                "fluid_overload_evidence_present",
            ],
        )

    if ctx.fluid_overload_evidence == NEGATIVE:
        return rule_result(
            rule_id="G05",
            name="fluid overload risk",
            status=STATUS_CONDITIONAL,
            reason=(
                "当前未记录容量过负荷证据，但高档补液仍需"
                "结合后续监测和患者心肺状态。"
                f"{per_kg_note}"
            ),
            triggered_conditions=[
                "high_fluid_action",
                "fluid_overload_evidence_absent",
            ],
        )

    return rule_result(
        rule_id="G05",
        name="fluid overload risk",
        status=STATUS_INSUFFICIENT,
        reason=(
            "当前动作属于高档补液，但缺少可靠的容量过负荷"
            "评估证据，不能确认继续补液的容量风险。"
            f"{per_kg_note}"
        ),
        triggered_conditions=[
            "high_fluid_action",
            "fluid_overload_evidence_unknown",
        ],
        missing_evidence=[
            "fluid_overload_evidence",
            "cardiopulmonary_volume_assessment",
        ],
    )


# ============================================================
# 15. G06：晶体液和升压药联合治疗
#     修订：不稳定休克背景 + 联合动作，证据充分时按文档
#     第六节第 4 级返回 SUPPORTED_BY_GUIDELINE（"不稳定休克
#     可同时进行晶体液与升压药支持"是明确的指南依据）。
# ============================================================

def assess_g06_combination(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    is_combination = (
        action.fluid_bin > 0
        and action.vaso_bin > 0
    )

    if not is_combination:
        return rule_result(
            rule_id="G06",
            name="combined fluid and vasopressor treatment",
            status=STATUS_CONDITIONAL,
            reason=(
                "当前动作不是同时包含补液和升压药的联合动作。"
            ),
            triggered_conditions=[
                "not_combined_action",
            ],
        )

    if ctx.shock_instability == POSITIVE:
        return rule_result(
            rule_id="G06",
            name="combined fluid and vasopressor treatment",
            status=STATUS_SUPPORTED,
            reason=(
                "当前记录提示存在不稳定休克背景，"
                "指南允许不稳定休克同时给予晶体液与升压药支持；"
                "当前规则库为联合动作找到了适用依据，"
                "但具体剂量仍不能由该规则直接确定。"
            ),
            triggered_conditions=[
                "combined_action",
                "shock_instability_present",
            ],
        )

    if ctx.shock_instability == NEGATIVE:
        return rule_result(
            rule_id="G06",
            name="combined fluid and vasopressor treatment",
            status=STATUS_CONDITIONAL,
            reason=(
                "当前没有明确不稳定休克证据；联合动作不能仅凭"
                "动作组合本身获得支持，需要重新核查 MAP、灌注"
                "和既往复苏情况。"
            ),
            triggered_conditions=[
                "combined_action",
                "shock_instability_not_present",
            ],
        )

    return rule_result(
        rule_id="G06",
        name="combined fluid and vasopressor treatment",
        status=STATUS_INSUFFICIENT,
        reason=(
            "当前动作同时包含补液和升压药，但缺少可靠的"
            "休克不稳定性证据，无法确认联合治疗背景。"
        ),
        triggered_conditions=[
            "combined_action",
            "shock_instability_unknown",
        ],
        missing_evidence=[
            "shock_instability",
        ],
    )


# ============================================================
# 16. 规则合并
#     严格按文档第六节优先级：
#     RULE_CONFLICT > INSUFFICIENT_INFORMATION > CONDITIONAL
#     > SUPPORTED_BY_GUIDELINE
# ============================================================

def merge_rule_statuses(
    assessments: Sequence[Dict[str, Any]],
) -> str:
    statuses = {
        assessment.get("status")
        for assessment in assessments
    }

    for status in STATUS_PRIORITY:
        if status in statuses:
            return status

    # 理论上不可达（每条规则都会返回四种状态之一）；
    # 兜底按最保守处理。
    return STATUS_INSUFFICIENT


def merge_reasons(
    assessments: Sequence[Dict[str, Any]],
) -> str:
    parts = []

    for assessment in assessments:
        rule_id = assessment.get("rule_id", "UNKNOWN")
        status = assessment.get("status", "UNKNOWN")
        reason = assessment.get("reason", "")

        parts.append(
            f"{rule_id}[{status}]: {reason}"
        )

    return "；".join(parts)


def evaluate_single_action(
    ctx: GuidelinePatientContext,
    action: TreatmentAction,
) -> Dict[str, Any]:
    assessments = [
        assess_g01_map_target(ctx, action),
        assess_g02_persistent_hypotension(ctx, action),
        assess_g03_fluid_responsiveness(ctx, action),
        assess_g04_cumulative_fluid(ctx, action),
        assess_g05_fluid_overload(ctx, action),
        assess_g06_combination(ctx, action),
    ]

    all_missing_evidence = sorted(
        {
            item
            for assessment in assessments
            for item in assessment.get(
                "missing_evidence",
                [],
            )
        }
    )

    all_triggered_conditions = sorted(
        {
            item
            for assessment in assessments
            for item in assessment.get(
                "triggered_conditions",
                [],
            )
        }
    )

    merged_status = merge_rule_statuses(assessments)

    return {
        "patient_id": ctx.patient_id,
        "time_index": ctx.time_index,
        "action_id": action.action_id,
        "fluid_bin": action.fluid_bin,
        "fluid_level": action.fluid_level,
        "vaso_bin": action.vaso_bin,
        "vaso_level": action.vaso_level,
        "merged_status": merged_status,
        "reason": merge_reasons(assessments),
        "triggered_conditions": all_triggered_conditions,
        "missing_evidence": all_missing_evidence,
        "rule_assessments": assessments,
        "context": asdict(ctx),
    }


def evaluate_all_actions(
    ctx: GuidelinePatientContext,
) -> List[Dict[str, Any]]:
    """
    文档规定的动作枚举：
    fluid_bin ∈ {0,1,2,3,4}
    vaso_bin ∈ {0,1,2,3,4}

    action_id = fluid_bin * 5 + vaso_bin
    """
    results = []

    for fluid_bin in range(5):
        for vaso_bin in range(5):
            action_id = fluid_bin * 5 + vaso_bin

            action = TreatmentAction(
                action_id=action_id,
                fluid_bin=fluid_bin,
                vaso_bin=vaso_bin,
            )

            results.append(
                evaluate_single_action(
                    ctx=ctx,
                    action=action,
                )
            )

    return results


# ============================================================
# 17. 输出格式
# ============================================================

def flatten_result_for_csv(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "patient_id": result.get("patient_id"),
        "time_index": result.get("time_index"),
        "action_id": result.get("action_id"),
        "fluid_bin": result.get("fluid_bin"),
        "fluid_level": result.get("fluid_level"),
        "vaso_bin": result.get("vaso_bin"),
        "vaso_level": result.get("vaso_level"),
        "merged_status": result.get("merged_status"),
        "reason": result.get("reason"),
        "triggered_conditions": json.dumps(
            result.get("triggered_conditions", []),
            ensure_ascii=False,
        ),
        "missing_evidence": json.dumps(
            result.get("missing_evidence", []),
            ensure_ascii=False,
        ),
    }

    context = result.get("context", {})

    for key, value in context.items():
        if isinstance(value, (list, dict)):
            row[f"context_{key}"] = json.dumps(
                value,
                ensure_ascii=False,
            )
        else:
            row[f"context_{key}"] = value

    rule_assessments = result.get(
        "rule_assessments",
        [],
    )

    for assessment in rule_assessments:
        rule_id = assessment.get("rule_id")

        if not rule_id:
            continue

        row[f"{rule_id}_status"] = assessment.get(
            "status"
        )
        row[f"{rule_id}_source"] = assessment.get(
            "source"
        )
        row[f"{rule_id}_reason"] = assessment.get(
            "reason"
        )
        row[f"{rule_id}_missing_evidence"] = json.dumps(
            assessment.get(
                "missing_evidence",
                [],
            ),
            ensure_ascii=False,
        )

    return row


def save_results(
    results: List[Dict[str, Any]],
    output_json: Path,
    output_csv: Path,
) -> None:
    with output_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    flattened = [
        flatten_result_for_csv(result)
        for result in results
    ]

    pd.DataFrame(flattened).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 18. 主流程
# ============================================================

def choose_decision_index(
    trajectory: pd.DataFrame,
    time_column: Optional[str],
    requested_time_index: Optional[str],
) -> int:
    if requested_time_index is None:
        return len(trajectory) - 1

    if time_column is None:
        raise ValueError(
            "指定了 --time-index，但 CSV 中没有识别到时间字段。"
        )

    requested_numeric = safe_float(requested_time_index)

    if requested_numeric is not None:
        numeric_time = pd.to_numeric(
            trajectory[time_column],
            errors="coerce",
        )

        matched = np.where(
            np.isclose(
                numeric_time.fillna(np.nan).to_numpy(),
                requested_numeric,
                equal_nan=False,
            )
        )[0]

        if len(matched) > 0:
            return int(matched[-1])

    matched_indices = [
        index
        for index, value in enumerate(
            trajectory[time_column].tolist()
        )
        if str(value) == str(requested_time_index)
    ]

    if matched_indices:
        return matched_indices[-1]

    raise ValueError(
        f"未找到时间点：{requested_time_index}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MIMIC CSV 指南约束模块："
            "对每个患者的 25 个治疗动作进行 G01-G06 评估。"
        )
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="MIMIC 提取 CSV 路径。",
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help=(
            "只评估指定患者。若不指定，则评估 CSV 中所有患者。"
        ),
    )

    parser.add_argument(
        "--time-index",
        type=str,
        default=None,
        help=(
            "指定决策时点。若不指定，则使用每个患者"
            "轨迹最后一个时点。"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="JSON 输出路径。",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="CSV 输出路径。",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    print(f"[INFO] 读取数据：{data_path}")

    df = read_csv_strict(data_path)

    print(f"[INFO] 数据行数：{len(df)}")
    print(f"[INFO] 数据列数：{len(df.columns)}")

    detected = detect_columns(df)

    print("[INFO] 自动识别字段：")
    for logical_name, column_name in detected.items():
        print(
            f"       {logical_name}: "
            f"{column_name if column_name else '未识别'}"
        )

    if detected.get("map") is None:
        print(
            "[WARNING] 未识别到 MAP 字段，G01/G02 将返回 "
            "INSUFFICIENT_INFORMATION。",
            file=sys.stderr,
        )

    if detected.get("lactate") is None:
        print(
            "[WARNING] 未识别到乳酸字段，休克不稳定性判断可能为 unknown。",
            file=sys.stderr,
        )

    if detected.get("patient_id") is None:
        print(
            "[WARNING] 未识别到患者 ID 字段，将整个 CSV 视作一条轨迹。",
            file=sys.stderr,
        )

    if ENABLE_RULE_CONFLICT:
        print(
            "[INFO] 规则冲突开关已开启，当前启用的待审核冲突定义："
            + json.dumps(RULE_CONFLICT_DEFINITIONS, ensure_ascii=False)
        )
    else:
        print("[INFO] 规则冲突开关已关闭，本轮不会产生 RULE_CONFLICT。")

    groups = get_patient_groups(
        df=df,
        detected=detected,
    )

    all_results: List[Dict[str, Any]] = []

    for patient_id, trajectory in groups:
        if args.patient_id is not None:
            if str(patient_id) != str(args.patient_id):
                continue

        decision_index = choose_decision_index(
            trajectory=trajectory,
            time_column=detected.get("time_index"),
            requested_time_index=args.time_index,
        )

        context = build_guideline_context(
            trajectory=trajectory,
            detected=detected,
            decision_index=decision_index,
        )

        action_results = evaluate_all_actions(context)
        all_results.extend(action_results)

        print(
            f"[INFO] 患者 {patient_id} "
            f"时点 {context.time_index} "
            f"已完成 25 个动作评估。"
        )

    if not all_results:
        raise ValueError(
            "没有生成任何结果。请检查 --patient-id 是否正确，"
            "或者检查 CSV 中是否存在患者记录。"
        )

    save_results(
        results=all_results,
        output_json=output_json,
        output_csv=output_csv,
    )

    print(f"[INFO] JSON 已保存：{output_json.resolve()}")
    print(f"[INFO] CSV 已保存：{output_csv.resolve()}")

    status_count = (
        pd.Series(
            [
                result["merged_status"]
                for result in all_results
            ]
        )
        .value_counts()
        .to_dict()
    )

    print("[INFO] 合并状态统计：")
    for status, count in status_count.items():
        print(f"       {status}: {count}")


if __name__ == "__main__":
    main()