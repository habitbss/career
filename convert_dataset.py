# convert_dataset.py
"""
把你下载的数据集，转换成 data/jobs.csv（我们项目要的 7 列）。

用法：
    python convert_dataset.py 原始数据.csv

支持你那个数据集的列名：
    jobname / company / salary / city / description / other / label
    minsalary / maxsalary / meansalary / city_idx

做的事：
    1. 列名映射
    2. 薪资：minsalary/maxsalary（单位是元）-> 千元，写成 "25-50K"
    3. 技能：没有 skills 列，所以从 description 里按词典抽取
    4. 学历/经验：从 description 里用正则抽
    5. 去重

输出的列：title,company,city,salary,education,experience,skills,description,url
"""

import argparse
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 技能词典：标准名 -> [各种写法]
# 想加技能就直接往这里加，加完重跑脚本
# ============================================================
SKILLS: dict[str, list[str]] = {
    # --- 数据工具 ---
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "Oracle": ["oracle"],
    "PostgreSQL": ["postgresql", "postgres", "pg数据库"],
    "Excel": ["excel"],
    "PPT": ["ppt", "powerpoint"],
    "Tableau": ["tableau"],
    "PowerBI": ["powerbi", "power bi"],
    "FineBI": ["finebi"],
    "帆软": ["帆软"],
    "SPSS": ["spss"],
    "SAS": ["sas"],
    "R语言": ["r语言"],
    # --- 大数据 ---
    "Hive": ["hive"],
    "Spark": ["spark"],
    "Hadoop": ["hadoop"],
    "Flink": ["flink"],
    "Kafka": ["kafka"],
    "HBase": ["hbase"],
    "ClickHouse": ["clickhouse"],
    "Doris": ["doris"],
    "StarRocks": ["starrocks"],
    "Presto": ["presto"],
    "Elasticsearch": ["elasticsearch", "es集群"],
    "Redis": ["redis"],
    "MongoDB": ["mongodb"],
    # --- 数据工程 ---
    "ETL": ["etl"],
    "数据仓库": ["数据仓库", "数仓"],
    "数据建模": ["数据建模", "维度建模", "数仓建模"],
    "数据湖": ["数据湖"],
    "OLAP": ["olap"],
    "数据治理": ["数据治理"],
    "数据质量": ["数据质量"],
    # --- 分析方法 ---
    "A/B测试": ["a/b测试", "ab测试", "a/b test", "abtest"],
    "漏斗分析": ["漏斗分析", "漏斗模型"],
    "用户画像": ["用户画像"],
    "归因分析": ["归因分析"],
    "竞品分析": ["竞品分析"],
    "数据挖掘": ["数据挖掘"],
    "统计分析": ["统计分析"],
    "回归分析": ["回归分析"],
    "聚类": ["聚类"],
    "时间序列": ["时间序列"],
    "指标体系": ["指标体系"],
    "埋点": ["埋点"],
    "数据可视化": ["数据可视化", "可视化"],
    "数据看板": ["看板", "dashboard"],
    "报表": ["报表"],
    "商业分析": ["商业分析"],
    "业务分析": ["业务分析"],
    # --- 编程 ---
    "Python": ["python"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Java": ["java"],
    "Scala": ["scala"],
    "Golang": ["golang", "go语言"],
    "C++": ["c++"],
    "Shell": ["shell"],
    "Linux": ["linux"],
    "Git": ["git"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    # --- 机器学习 ---
    "机器学习": ["机器学习"],
    "深度学习": ["深度学习"],
    "NLP": ["nlp", "自然语言处理"],
    "计算机视觉": ["计算机视觉", "cv算法"],
    "推荐系统": ["推荐系统", "推荐算法"],
    "特征工程": ["特征工程"],
    "XGBoost": ["xgboost"],
    "LightGBM": ["lightgbm"],
    "随机森林": ["随机森林"],
    "SVM": ["svm", "支持向量机"],
    "神经网络": ["神经网络"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Keras": ["keras"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "大模型": ["大模型", "llm"],
    "Transformer": ["transformer"],
    "RAG": ["rag"],
    # # --- 软技能/业务 ---
    # "需求分析": ["需求分析"],
    # "项目管理": ["项目管理"],
    # "团队协作": ["团队协作"],
    # "沟通": ["沟通"],
}

# 中文词直接查子串；英文词要求词边界，避免 sql 命中 mysql
_ENGLISH = re.compile(r"[a-z0-9+#./]+")


def _build_patterns():
    en, zh = [], []
    for canonical, aliases in SKILLS.items():
        for alias in aliases + [canonical]:
            a = alias.lower().strip()
            if not a:
                continue
            if _ENGLISH.fullmatch(a):
                pat = re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])")
                en.append((canonical, pat))
            else:
                zh.append((canonical, a))
    return en, zh


EN_PATTERNS, ZH_TERMS = _build_patterns()

EDU_PAT = re.compile(r"(本科|硕士|博士|大专|专科|研究生)")
# 覆盖：3-5年 / 3~5年 / 5年以上 / 5年及以上 / 5年以内
EXP_PAT = re.compile(
    r"(\d+)\s*[-~到至]\s*(\d+)\s*年"
    r"|(\d+)\s*年\s*(?:及)?以?上"
    r"|(\d+)\s*年\s*(?:及)?以?内"
)


def extract_skills(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    low = text.lower()
    found: list[str] = []
    for canonical, pat in EN_PATTERNS:
        if pat.search(low):
            found.append(canonical)
    for canonical, term in ZH_TERMS:
        if term in text:
            found.append(canonical)
    return list(dict.fromkeys(found))


def extract_education(text: str) -> str:
    m = EDU_PAT.search(text or "")
    if not m:
        return ""
    return {"专科": "大专", "研究生": "硕士"}.get(m.group(1), m.group(1))


def extract_experience(text: str) -> str:
    m = EXP_PAT.search(text or "")
    if not m:
        return ""
    if m.group(1) and m.group(2):
        return f"{m.group(1)}-{m.group(2)}年"
    if m.group(3):
        return f"{m.group(3)}年以上"
    if m.group(4):
        return f"{m.group(4)}年以内"
    return ""


def to_k(value):
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v) or v <= 0:
        return None
    return int(round(v / 1000))


def read_table(path: Path):
    """自动识别 .csv / .xlsx，返回 (DataFrame, 说明)。"""
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise SystemExit(
                "\n需要先装一个读取 Excel 的库，运行这条命令：\n"
                "    pip install openpyxl\n"
                "装完再跑一次。\n"
                "（不想装的话：用 Excel 打开 → 另存为 → CSV UTF-8，然后传 csv 文件）\n")
        if suffix == ".xls":
            try:
                import xlrd  # noqa: F401
            except ImportError:
                raise SystemExit(
                    "\n.xls 老格式需要 xlrd：pip install xlrd\n"
                    "更方便的做法：用 Excel 另存为 .xlsx 或 CSV UTF-8。\n")
        book = pd.ExcelFile(path)
        sheet = book.sheet_names[0]
        print(f"[读取] {path.name}  工作表 {book.sheet_names}  默认用第一个：{sheet}")
        return book.parse(sheet, dtype=str).fillna("")

    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str).fillna("")
            print(f"[读取] {path.name}  编码={enc}  行数={len(df)}")
            return df
        except UnicodeDecodeError:
            continue

    raise SystemExit("文件编码认不出来。用 Excel 打开后「另存为 → CSV UTF-8」再试。")


def main(raw_path: str, out_path: str):
    raw = Path(raw_path)
    if not raw.exists():
        raise FileNotFoundError(f"找不到文件：{raw}")

    df = read_table(raw)
    print(f"[读取] 行数={len(df)}")

    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[列名] {list(df.columns)}")

    # 列名兼容：不同数据集叫法不一样
    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_title = col("jobname", "job_name", "title", "职位名称", "岗位名称")
    c_company = col("company", "公司名称", "公司")
    c_city = col("city", "城市", "工作城市", "location")
    c_desc = col("description", "jd", "job_description", "岗位描述")
    c_edu = col("education", "学历")
    c_exp = col("experience", "经验", "工作经验")
    c_min = col("minsalary", "min_salary")
    c_max = col("maxsalary", "max_salary")
    c_salary = col("salary", "薪资")

    for name, c in [("岗位名", c_title), ("公司", c_company), ("城市", c_city)]:
        if not c:
            raise ValueError(f"原始数据里找不到「{name}」列，实际列名：{list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        title = str(r.get(c_title, "")).strip()
        company = str(r.get(c_company, "")).strip()
        city = str(r.get(c_city, "")).strip()
        desc = str(r.get(c_desc, "")) if c_desc else ""
        if not title or not company or not city:
            continue

        # 薪资：优先用 minsalary/maxsalary（元 -> 千元）
        lo = to_k(r.get(c_min)) if c_min else None
        hi = to_k(r.get(c_max)) if c_max else None
        if lo is None and c_salary:
            # 兜底：从 "(25000.0, 50000.0)" 这种字符串里抠数字
            nums = re.findall(r"\d+(?:\.\d+)?", str(r.get(c_salary, "")))
            nums = [float(x) for x in nums if float(x) > 100]
            if nums:
                lo, hi = int(round(min(nums) / 1000)), int(round(max(nums) / 1000))

        salary = ""
        if lo and hi:
            lo, hi = min(lo, hi), max(lo, hi)
            salary = f"{lo}-{hi}K"

        skills = extract_skills(f"{title} {desc}")

        rows.append({
            "title": title,
            "company": company,
            "city": city,
            "salary": salary,
            "education": extract_education(desc) or (str(r.get(c_edu, "")).strip() if c_edu else ""),
            "experience": extract_experience(desc) or (str(r.get(c_exp, "")).strip() if c_exp else ""),
            "skills": ";".join(skills),
            "description": desc[:2000],          # 截断，避免文件过大
            "url": "",
        })

    out = pd.DataFrame(rows).drop_duplicates(subset=["title", "company", "city"])
    before = len(out)
    out = out[out["skills"].str.len() > 0]

    DATA_DIR.mkdir(exist_ok=True)
    out_path = Path(out_path)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[输出] {out_path}")
    print(f"  有效岗位：{len(out)} 条（已去重、已丢弃无技能的岗位）")
    print(f"  因抽不出技能被丢弃：{before - len(out)} 条  "
          f"（占比 {(before - len(out)) / max(before, 1):.1%}，太高说明技能词典不够）")
    print(f"  平均每条技能数：{out['skills'].str.count(';').add(1).mean():.1f}")

    city_top = out["city"].value_counts().head(5).to_dict()
    print(f"  城市 Top5：{city_top}")

    skill_counter: dict[str, int] = {}
    for s in out["skills"]:
        for x in s.split(";"):
            if x:
                skill_counter[x] = skill_counter.get(x, 0) + 1
    print(f"  技能 Top10：{dict(sorted(skill_counter.items(), key=lambda kv: -kv[1])[:10])}")
    print(f"\n[下一步] 检查 data/jobs.csv 没问题后，运行：python kg_builder.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="原始数据文件路径")
    ap.add_argument("-o", "--out", default=str(DATA_DIR / "jobs.csv"), help="输出路径")
    args = ap.parse_args()
    main(args.raw, args.out)
