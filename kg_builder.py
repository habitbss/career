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


# ---------- 1. 数据获取（模拟爬虫） ----------
def crawl_jobs():
    jobs = [
        {"title": "Python后端工程师",
         "skills": ["Python", "Django", "MySQL", "Redis", "Linux"],
         "city": "北京", "company": "字节跳动",
         "salary_min": 25, "salary_max": 45,
         "education": "本科", "experience": "3-5年"},
        {"title": "Java开发工程师",
         "skills": ["Java", "Spring", "MySQL", "Kafka", "微服务"],
         "city": "上海", "company": "阿里巴巴",
         "salary_min": 28, "salary_max": 50,
         "education": "本科", "experience": "3-5年"},
        {"title": "算法工程师",
         "skills": ["Python", "机器学习", "深度学习", "PyTorch", "数学"],
         "city": "北京", "company": "百度",
         "salary_min": 35, "salary_max": 70,
         "education": "硕士", "experience": "3-5年"},
        {"title": "数据分析师",
         "skills": ["Python", "SQL", "Pandas", "统计学", "Excel"],
         "city": "北京", "company": "美团",
         "salary_min": 18, "salary_max": 30,
         "education": "本科", "experience": "1-3年"},
        {"title": "前端开发工程师",
         "skills": ["JavaScript", "Vue", "React", "HTML", "CSS"],
         "city": "深圳", "company": "腾讯",
         "salary_min": 20, "salary_max": 40,
         "education": "本科", "experience": "1-3年"},
        {"title": "大数据开发工程师",
         "skills": ["Java", "Spark", "Hadoop", "SQL", "Kafka"],
         "city": "杭州", "company": "阿里巴巴",
         "salary_min": 25, "salary_max": 45,
         "education": "本科", "experience": "3-5年"},
        {"title": "运维开发工程师",
         "skills": ["Linux", "Python", "Docker", "Kubernetes", "Shell"],
         "city": "杭州", "company": "网易",
         "salary_min": 20, "salary_max": 38,
         "education": "本科", "experience": "3-5年"},
        {"title": "测试开发工程师",
         "skills": ["Python", "Selenium", "SQL", "Linux", "自动化测试"],
         "city": "上海", "company": "美团",
         "salary_min": 18, "salary_max": 32,
         "education": "本科", "experience": "1-3年"},
        {"title": "数据产品经理",
         "skills": ["SQL", "数据分析", "产品设计", "沟通", "Axure"],
         "city": "北京", "company": "京东",
         "salary_min": 20, "salary_max": 38,
         "education": "本科", "experience": "3-5年"},
    ]
    noisy = []
    for j in jobs:
        j2 = dict(j)
        j2["title"] = " " + j2["title"] + " "
        if random.random() < 0.2:
            j2["salary_max"] = None
        noisy.append(j2)
    noisy.append(noisy[0])

    (DATA_DIR / "raw_jobs.json").write_text(
        json.dumps(noisy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[1] 数据获取：{len(noisy)} 条原始数据")
    return noisy


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
