# convert_dataset.py
"""
把你下载的数据集，转换成 data/jobs.csv（我们项目要的 9 列）。

用法：
    python convert_dataset.py data/raw.xlsx

做的事：
    1. 列名映射（jobname->title 等）
    2. 薪资：minsalary/maxsalary（元）-> 千元，写成 "25-50K"
    3. 技能：从 description 里按词典抽取（本文件自带 140+ 个技能词）
    4. 学历/经验：从 description 里用正则抽
    5. 去重

想加技能：往下面的 SKILLS 字典里加一行，格式 "标准名": ["写法1", "写法2"]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# 一个岗位至少要抽到几个技能才保留。
# 只抽到 1 个技能的岗位（多是生物、金融等非技术岗），留在推荐系统里就是噪音。
MIN_SKILLS = 2

# ============================================================
# 技能词典：标准名 -> [各种写法]（小写）
# ============================================================
SKILLS: dict[str, list[str]] = {
    # ---------- 编程语言 ----------
    "Python": ["python"],
    "Java": ["java"],
    "Golang": ["golang", "go语言"],
    "C++": ["c++"],
    "C#": ["c#"],
    "PHP": ["php"],
    "Scala": ["scala"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "R语言": ["r语言"],
    "MATLAB": ["matlab"],
    "Shell": ["shell", "bash"],
    "SQL": ["sql"],

    # ---------- 前端 ----------
    "HTML": ["html"],
    "CSS": ["css"],
    "Vue": ["vue"],
    "React": ["react"],
    "Angular": ["angular"],
    "Node.js": ["node.js", "nodejs"],
    "小程序": ["小程序"],
    "Flutter": ["flutter"],
    "React Native": ["react native"],

    # ---------- 后端 / 架构 ----------
    "Spring": ["spring"],
    "SpringBoot": ["springboot", "spring boot"],
    "SpringCloud": ["springcloud", "spring cloud"],
    "MyBatis": ["mybatis"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Netty": ["netty"],
    "微服务": ["微服务"],
    "分布式": ["分布式"],
    "高并发": ["高并发"],
    "消息队列": ["消息队列"],
    "RocketMQ": ["rocketmq"],
    "RabbitMQ": ["rabbitmq"],
    "Nginx": ["nginx"],

    # ---------- 数据库 ----------
    "MySQL": ["mysql"],
    "Oracle": ["oracle"],
    "PostgreSQL": ["postgresql", "postgres"],
    "SQLServer": ["sqlserver", "sql server"],
    "MongoDB": ["mongodb"],
    "Redis": ["redis"],
    "HBase": ["hbase"],
    "ClickHouse": ["clickhouse"],
    "Doris": ["doris"],
    "Elasticsearch": ["elasticsearch"],
    "TiDB": ["tidb"],

    # ---------- 大数据 ----------
    "Hadoop": ["hadoop"],
    "Spark": ["spark"],
    "Flink": ["flink"],
    "Hive": ["hive"],
    "Kafka": ["kafka"],
    "数据仓库": ["数据仓库", "数仓"],
    "ETL": ["etl"],
    "数据湖": ["数据湖"],
    "实时计算": ["实时计算"],
    "离线计算": ["离线计算", "离线数仓"],

    # ---------- AI / 算法 ----------
    "机器学习": ["机器学习"],
    "深度学习": ["深度学习"],
    "NLP": ["nlp", "自然语言处理"],
    "计算机视觉": ["计算机视觉", "cv算法"],
    "推荐系统": ["推荐系统", "推荐算法"],
    "搜索算法": ["搜索算法", "搜索推荐"],
    "排序算法": ["排序算法", "排序模型", "ctr预估"],
    "特征工程": ["特征工程"],
    "XGBoost": ["xgboost"],
    "LightGBM": ["lightgbm"],
    "随机森林": ["随机森林"],
    "SVM": ["svm", "支持向量机"],
    "聚类": ["聚类"],
    "回归分析": ["回归分析"],
    "时间序列": ["时间序列"],
    "强化学习": ["强化学习"],
    "知识图谱": ["知识图谱"],
    "大模型": ["大模型", "llm"],
    "Transformer": ["transformer"],
    "RAG": ["rag"],
    "模型微调": ["微调", "fine-tuning", "finetune"],
    "Prompt": ["prompt工程", "提示词"],
    "AIGC": ["aigc"],
    "多模态": ["多模态"],
    "OCR": ["ocr"],
    "语音识别": ["语音识别"],
    "神经网络": ["神经网络"],
    "模型部署": ["模型部署", "模型压缩"],

    # ---------- 框架 / 工具 ----------
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Keras": ["keras"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Matplotlib": ["matplotlib"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Jenkins": ["jenkins"],
    "Git": ["git"],
    "Linux": ["linux"],

    # ---------- 云 ----------
    "AWS": ["aws"],
    "阿里云": ["阿里云"],
    "腾讯云": ["腾讯云"],
    "华为云": ["华为云"],

    # ---------- 测试 ----------
    "自动化测试": ["自动化测试"],
    "接口测试": ["接口测试"],
    "性能测试": ["性能测试"],
    "Selenium": ["selenium"],
    "Appium": ["appium"],
    "JMeter": ["jmeter"],
    "Postman": ["postman"],
    "单元测试": ["单元测试"],

    # ---------- 运维 ----------
    "运维": ["运维"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "Ansible": ["ansible"],
    "CICD": ["ci/cd", "cicd", "持续集成"],

    # ---------- 数据 / BI ----------
    "数据可视化": ["数据可视化"],
    "报表": ["报表"],
    "Tableau": ["tableau"],
    "PowerBI": ["powerbi", "power bi"],
    "FineBI": ["finebi"],
    "指标体系": ["指标体系"],
    "埋点": ["埋点"],
    "A/B测试": ["a/b测试", "ab测试", "a/b test", "abtest"],
    "用户画像": ["用户画像"],
    "数据挖掘": ["数据挖掘"],
    "统计分析": ["统计分析"],
    "漏斗分析": ["漏斗分析"],
    "归因分析": ["归因分析"],
    "数据治理": ["数据治理"],
    "数据质量": ["数据质量"],
    "OLAP": ["olap"],
    "Excel": ["excel"],
    "PPT": ["ppt", "powerpoint"],
    "SPSS": ["spss"],
    "SAS": ["sas"],
    "数据建模": ["数据建模", "维度建模"],

    # ---------- 金融 / 区块链 ----------
    "风控": ["风控"],
    "量化": ["量化"],
    "反欺诈": ["反欺诈"],
    "信贷": ["信贷"],
    "区块链": ["区块链"],
    "智能合约": ["智能合约"],
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
    """自动识别 .csv / .xlsx，返回 DataFrame。"""
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise SystemExit(
                "\n需要先装一个读取 Excel 的库，运行这条命令：\n"
                "    pip install openpyxl\n"
                "装完再跑一次。\n")
        if suffix == ".xls":
            try:
                import xlrd  # noqa: F401
            except ImportError:
                raise SystemExit("\n.xls 老格式需要 xlrd：pip install xlrd\n"
                                 "更方便：用 Excel 另存为 .xlsx。\n")
        book = pd.ExcelFile(path)
        sheet = book.sheet_names[0]
        print(f"[读取] {path.name}  工作表 {book.sheet_names}  用第一个：{sheet}")
        return book.parse(sheet, dtype=str).fillna("")

    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str).fillna("")
            print(f"[读取] {path.name}  编码={enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise SystemExit("文件编码认不出来。用 Excel 另存为 CSV UTF-8 再试。")


def main(raw_path: str, out_path: str):
    raw = Path(raw_path)
    if not raw.exists():
        raise FileNotFoundError(
            f"找不到文件：{raw}\n"
            f"当前所在目录：{Path.cwd()}\n"
            "如果文件在 data 文件夹里，命令要写成：python convert_dataset.py data/文件名.xlsx")

    df = read_table(raw)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[读取] 行数={len(df)}")
    print(f"[列名] {list(df.columns)}")

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
            raise SystemExit(f"原始数据里找不到「{name}」列，实际列名：{list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        title = str(r.get(c_title, "")).strip()
        company = str(r.get(c_company, "")).strip()
        city = str(r.get(c_city, "")).strip()
        desc = str(r.get(c_desc, "")) if c_desc else ""
        if not title or not company or not city:
            continue

        lo = to_k(r.get(c_min)) if c_min else None
        hi = to_k(r.get(c_max)) if c_max else None
        if lo is None and c_salary:
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(r.get(c_salary, "")))
                    if float(x) > 100]
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
            "description": desc[:2000],
            "url": "",
        })

    out = pd.DataFrame(rows).drop_duplicates(subset=["title", "company", "city"])
    before = len(out)
    n_skills = out["skills"].apply(lambda s: len([x for x in s.split(";") if x]))
    out = out[n_skills >= MIN_SKILLS]

    DATA_DIR.mkdir(exist_ok=True)
    out_path = Path(out_path)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    skill_counts = out["skills"].str.count(";").add(1)
    print(f"\n[输出] {out_path}")
    print(f"  有效岗位：{len(out)} 条")
    print(f"  因抽不出技能被丢弃：{before - len(out)} 条 "
          f"（占比 {(before - len(out)) / max(before, 1):.1%}）")
    print(f"  平均每条技能数：{skill_counts.mean():.1f}  （越多区分度越好）")
    print(f"  只有 1 个技能的岗位：{(skill_counts == 1).sum()} 条 "
          f"（占 {(skill_counts == 1).mean():.1%}，太多说明词典还不够）")
    print(f"  城市 Top5：{out['city'].value_counts().head(5).to_dict()}")

    counter: dict[str, int] = {}
    for s in out["skills"]:
        for x in s.split(";"):
            if x:
                counter[x] = counter.get(x, 0) + 1
    print(f"  技能 Top12：{dict(sorted(counter.items(), key=lambda kv: -kv[1])[:12])}")
    print(f"\n[下一步] python kg_builder.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="原始数据文件路径")
    ap.add_argument("-o", "--out", default=str(DATA_DIR / "jobs.csv"), help="输出路径")
    args = ap.parse_args()
    main(args.raw, args.out)
