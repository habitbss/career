# normalize.py
"""
归一化：把"人写的、乱七八糟的"文本变成"机器能比的"标准值。

新增文件，不改动你原来的任何代码，只在 kg_builder 里调用。

三件事：
1. 薪资：统一成 K/月 的整数区间
2. 城市：北京/北京市/北京·朝阳 -> 北京
3. 技能：同义词映射到唯一标准名（表在 data/skills_synonyms.csv）
"""

import re
from pathlib import Path

import pandas as pd

DEFAULT_SYNONYM_PATH = Path(__file__).parent / "data" / "skills_synonyms.csv"

# ---------- 城市 ----------
CITY_MAP = {
    "北京市": "北京", "上海市": "上海", "深圳市": "深圳", "广州市": "广州",
    "杭州市": "杭州", "南京市": "南京", "成都市": "成都", "武汉市": "武汉",
    "西安市": "西安", "苏州市": "苏州", "长沙市": "长沙", "厦门市": "厦门",
}


def normalize_city(text) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.split(r"[·\-—/,，]", s)[0].strip()
    s = CITY_MAP.get(s, s)
    if s.endswith("市") and len(s) > 2:
        s = s[:-1]
    return s


# ---------- 薪资 ----------
_NUM = re.compile(r"\d+(?:\.\d+)?")


def parse_salary(text):
    """
    把各种薪资写法统一成 (min, max)，单位 K/月。
    支持：25-45K / 28-50K·14薪 / 1.5-2.5万 / 8000-12000元/月 / 20K / 面议
    面议类返回 (None, None)。
    """
    s = str(text or "").strip()
    if not s:
        return None, None
    if any(k in s for k in ["面议", "不限", "暂无", "保密"]):
        return None, None

    s = re.split(r"[·×*]", s)[0]

    if "年薪" in s:
        nums = [float(x) for x in _NUM.findall(s)]
        if not nums:
            return None, None
        lo = nums[0] * 10 / 12
        hi = (nums[1] if len(nums) > 1 else nums[0]) * 10 / 12
        return round(lo), round(hi)

    is_wan = "万" in s
    s = s.replace("元/月", "").replace("元", "").replace("/月", "")
    s = s.replace("月薪", "").replace("K", "").replace("k", "").replace("千", "")
    s = s.replace("万", "")

    nums = [float(x) for x in _NUM.findall(s)]
    if not nums:
        return None, None

    if is_wan:
        nums = [n * 10 for n in nums]
    elif max(nums) >= 1000:
        nums = [n / 1000 for n in nums]

    lo, hi = min(nums[0], nums[-1]), max(nums[0], nums[-1])
    return round(lo), round(hi)


# ---------- 技能 ----------
class SkillDict:
    """
    同义词字典。CSV 格式：alias, canonical, category, parent
    例如：k8s, Kubernetes, 云原生, 云原生
    """

    def __init__(self, alias_map=None):
        self.alias_map = alias_map or {}

    @classmethod
    def load(cls, path=DEFAULT_SYNONYM_PATH):
        path = Path(path)
        if not path.exists():
            print(f"[warn] 找不到同义词表 {path}，技能将不做归一")
            return cls()
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        m = {}
        for _, r in df.iterrows():
            alias, canonical = r["alias"].strip(), r["canonical"].strip()
            if alias and canonical:
                m[alias.lower()] = canonical
        return cls(m)

    def normalize(self, raw) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        return self.alias_map.get(s.lower(), s)
