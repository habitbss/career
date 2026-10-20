# kg_builder.py
"""
数据获取 -> 清洗 -> 归一 -> 三元组 -> 知识图谱
运行一次：python kg_builder.py

改造说明（相对你原来的版本）：
  1. crawl_jobs()   -> load_jobs()：从 data/jobs.csv 读真实数据，不再硬编码
  2. clean_jobs()   -> 增加技能同义词归一、薪资统一单位
  3. build_triples()-> requires 边上带 weight（前 5 个必备 1.0，其余加分 0.6）
  4. build_kg()     -> 把权重写进边
  其余结构没变，app.py / llm_qa.py 都能继续用。
"""

import pickle
import re
from pathlib import Path

import networkx as nx
import pandas as pd

from normalize import SkillDict, normalize_city, parse_salary

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

JOBS_CSV = DATA_DIR / "jobs.csv"
REQUIRED_COLS = ["title", "company", "city"]


# ---------- 1. 数据获取（从 CSV 读真实数据） ----------
def load_jobs():
    if not JOBS_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {JOBS_CSV}\n"
            "请把真实岗位数据放到 data/jobs.csv，表头至少包含：\n"
            "  title,company,city,salary,education,experience,skills\n"
            "（skills 用分号分隔，例如 Python;SQL;MySQL）"
        )

    df = pd.read_csv(JOBS_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"data/jobs.csv 缺少必备列 {missing}，实际列名：{list(df.columns)}")

    jobs = []
    for _, r in df.iterrows():
        smin, smax = parse_salary(r.get("salary", ""))
        skills = [s.strip() for s in re.split(r"[;；,，、/]", r.get("skills", ""))
                  if s.strip()]
        jobs.append({
            "title": r.get("title", ""),
            "company": r.get("company", ""),
            "city": r.get("city", ""),
            "salary_min": smin,
            "salary_max": smax,
            "education": r.get("education", ""),
            "experience": r.get("experience", ""),
            "skills": skills,
        })

    print(f"[1] 数据获取：{len(jobs)} 条原始数据（来自 {JOBS_CSV.name}）")
    if len(jobs) < 100:
        print(f"    [警告] 只有 {len(jobs)} 条，研一项目建议 >= 1000 条真实数据")
    return jobs


# ---------- 2. 数据处理 ----------
def clean_jobs(raw_jobs):
    sd = SkillDict.load(DATA_DIR / "skills_synonyms.csv")

    df = pd.DataFrame(raw_jobs)
    df["title"] = df["title"].str.strip()
    df["company"] = df["company"].str.strip()
    df["city"] = df["city"].map(normalize_city)

    # 缺关键字段的直接丢
    df = df[(df["title"].str.len() > 0)
            & (df["company"].str.len() > 0)
            & (df["city"].str.len() > 0)]
    df = df.drop_duplicates(subset=["title", "company", "city"])

    # 薪资统一成 K/月；面议记为 0，展示时再转成"面议"
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce").fillna(0).astype(int)
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce").fillna(0).astype(int)
    bad = df["salary_max"] < df["salary_min"]
    df.loc[bad, "salary_max"] = df.loc[bad, "salary_min"]

    # 技能归一：保留原始顺序（顺序 = 招聘方写的先后，用来定权重），顺带去重
    def _norm_skills(xs):
        out = []
        for x in xs:
            s = sd.normalize(x)
            if s and s not in out:
                out.append(s)
        return out

    df["skills"] = df["skills"].apply(_norm_skills)
    df = df[df["skills"].map(len) > 0]

    df.to_csv(DATA_DIR / "clean_jobs.csv", index=False, encoding="utf-8-sig")
    print(f"[2] 数据处理：{len(df)} 条清洗后数据")
    return df


# ---------- 3. 结构化（三元组） ----------
def build_triples(df):
    triples = []
    for _, row in df.iterrows():
        # 用「岗位名@公司」做节点 ID：否则几百个同名"数据分析师"会被合并成一个节点，
        # 公司、城市、薪资互相覆盖。
        job = f"job:{row['title']}@{row['company']}"
        triples.append((job, "offered_by", f"company:{row['company']}", {}))
        triples.append((job, "located_in", f"city:{row['city']}", {}))
        if row["education"]:
            triples.append((job, "requires_edu", f"edu:{row['education']}", {}))
        if row["experience"]:
            triples.append((job, "requires_exp", f"exp:{row['experience']}", {}))
        triples.append((job, "salary_range",
                        f"{row['salary_min']}-{row['salary_max']}",
                        {"min": int(row["salary_min"]), "max": int(row["salary_max"])}))

        for i, s in enumerate(row["skills"]):
            # 前 5 个技能视为必备(1.0)，后面的视为加分(0.6)
            w = 1.0 if i < 5 else 0.6
            triples.append((job, "requires", f"skill:{s}", {"weight": w}))

    pd.DataFrame(triples, columns=["head", "relation", "tail", "props"]).to_csv(
        DATA_DIR / "triples.csv", index=False, encoding="utf-8-sig")
    print(f"[3] 结构化：{len(triples)} 条三元组")
    return triples


# ---------- 4. 知识图谱 ----------
def build_kg(triples):
    G = nx.DiGraph()
    for h, r, t, props in triples:
        if not G.has_node(h):
            G.add_node(h, type=h.split(":", 1)[0])
        if not G.has_node(t):
            G.add_node(t, type=t.split(":", 1)[0])
        if r == "salary_range":
            G.nodes[h]["salary_min"] = props["min"]
            G.nodes[h]["salary_max"] = props["max"]
        G.add_edge(h, t, rel=r, **props)
    print(f"[4] 图谱：{G.number_of_nodes()} 节点 / {G.number_of_edges()} 边")
    return G


def build_and_save():
    raw = load_jobs()
    df = clean_jobs(raw)
    triples = build_triples(df)
    G = build_kg(triples)
    with open(DATA_DIR / "kg.pkl", "wb") as f:
        pickle.dump(G, f)
    print(f"[完成] 图谱已保存到 data/kg.pkl")
    return G


if __name__ == "__main__":
    build_and_save()
