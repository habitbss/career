# kg_builder.py
"""
数据获取 -> 清洗 -> 三元组 -> 知识图谱
运行一次：python kg_builder.py
"""
import json
import random
import pickle
from pathlib import Path
import pandas as pd
import networkx as nx

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

import re


def parse_salary(text):
    """把 "25-45K" / "1.5-2.5万" / "8000-12000元/月" / "面议" 统一成 (min, max)，单位 K/月。"""
    s = str(text or "").strip()
    if not s or any(k in s for k in ["面议", "不限", "暂无", "保密"]):
        return None, None
    s = re.split(r"[·×*]", s)[0]
    is_wan = "万" in s
    s = (s.replace("元/月", "").replace("元", "").replace("/月", "")
          .replace("月薪", "").replace("K", "").replace("k", "")
          .replace("千", "").replace("万", ""))
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None, None
    if is_wan:
        nums = [n * 10 for n in nums]
    elif max(nums) >= 1000:
        nums = [n / 1000 for n in nums]
    lo, hi = min(nums[0], nums[-1]), max(nums[0], nums[-1])
    return round(lo), round(hi)

# ---------- 1. 数据获取（模拟爬虫） ----------
def crawl_jobs():
    """从 data/jobs.csv 读取岗位数据（不再硬编码）。"""
    path = DATA_DIR / "jobs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n"
            "请把岗位数据放到 data/jobs.csv，表头：\n"
            "  title,company,city,salary,education,experience,skills")

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in ["title", "company", "city"] if c not in df.columns]
    if missing:
        raise ValueError(f"data/jobs.csv 缺少列 {missing}，实际列名：{list(df.columns)}")

    jobs = []
    for _, r in df.iterrows():
        smin, smax = parse_salary(r.get("salary", ""))
        jobs.append({
            "title": r.get("title", ""),
            "company": r.get("company", ""),
            "city": r.get("city", ""),
            "salary_min": smin,
            "salary_max": smax,
            "education": r.get("education", ""),
            "experience": r.get("experience", ""),
            "skills": [s.strip() for s in
                       str(r.get("skills", "")).replace("；", ";").split(";")
                       if s.strip()],
        })

    (DATA_DIR / "raw_jobs.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[1] 数据获取：{len(jobs)} 条原始数据（来自 jobs.csv）")
    return jobs

# ---------- 2. 数据处理 ----------
def clean_jobs(raw_jobs):
    df = pd.DataFrame(raw_jobs)
    df["title"] = df["title"].str.strip()
    df["company"] = df["company"].str.strip()
    df["city"] = df["city"].str.strip()
    df = df.drop_duplicates(subset=["title", "company", "city"])
    df["salary_min"] = df["salary_min"].fillna(0).astype(int)
    df["salary_max"] = df["salary_max"].fillna(df["salary_min"]).astype(int)
    df["skills"] = df["skills"].apply(
        lambda xs: sorted({x.strip() for x in xs if x and x.strip()}))
    df.to_csv(DATA_DIR / "clean_jobs.csv", index=False, encoding="utf-8-sig")
    print(f"[2] 数据处理：{len(df)} 条清洗后数据")
    return df


# ---------- 3. 结构化（三元组） ----------
def build_triples(df):
    triples = []
    for _, row in df.iterrows():
        job = f"job:{row['title']}"
        triples.append((job, "offered_by", f"company:{row['company']}", {}))
        triples.append((job, "located_in", f"city:{row['city']}", {}))
        triples.append((job, "requires_edu", f"edu:{row['education']}", {}))
        triples.append((job, "requires_exp", f"exp:{row['experience']}", {}))
        triples.append((job, "salary_range",
                        f"{row['salary_min']}-{row['salary_max']}",
                        {"min": int(row["salary_min"]), "max": int(row["salary_max"])}))
        for s in row["skills"]:
            triples.append((job, "requires", f"skill:{s}", {}))
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
        G.add_edge(h, t, rel=r)
    print(f"[4] 图谱：{G.number_of_nodes()} 节点 / {G.number_of_edges()} 边")
    return G


def build_and_save():
    raw = crawl_jobs()
    df = clean_jobs(raw)
    triples = build_triples(df)
    G = build_kg(triples)
    with open(DATA_DIR / "kg.pkl", "wb") as f:
        pickle.dump(G, f)
    print(f"[完成] 图谱已保存到 data/kg.pkl")
    return G


if __name__ == "__main__":
    build_and_save()
# -*-coding:utf-8-*-
