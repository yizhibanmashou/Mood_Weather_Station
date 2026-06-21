"""
Shared configuration for Mood Weather Station pipeline scripts.
Centralizes constants duplicated across 8+ scripts.
"""
from pathlib import Path

# Standard 34 Chinese provinces / municipalities / SARs
VALID_PROVINCES = {
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "四川", "贵州", "云南", "西藏",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
    "内蒙古", "香港", "澳门", "台湾",
}

# Six emotion dimensions (Ekman-based)
EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

EMOTION_CN = {
    "joy": "喜悦", "sadness": "悲伤", "anger": "愤怒",
    "fear": "恐惧", "surprise": "惊讶", "neutral": "中性",
}

EMOTION_COLORS = {
    "joy": "#FFD93D",
    "sadness": "#6C5CE7",
    "anger": "#E74C3C",
    "fear": "#2D3436",
    "surprise": "#00B894",
    "neutral": "#BDC3C7",
}

RANDOM_SEED = 42
