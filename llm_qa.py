# llm_qa.py
"""
LLM + 知识图谱 问答核心逻辑
"""
import os
import json
from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
USE_LLM = bool(DEEPSEEK_API_KEY)

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
) if USE_LLM else None


# ============================================================
# LLM 封装
# ============================================================
class LLM:
    def __init__(self, client):
        self.client = client

    def _chat(self, system, user, json_mode=False):
        if not self.client:
            return None
        kwargs = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[LLM 调用失败，降级] {e}")
            return None

    def parse_intent(self, query, all_skills, all_cities, all_companies, all_jobs):
        system = "你是一个招聘问答系统的语义解析器，只输出 JSON，不要任何解释。"
        user = f"""用户的提问：{query}

已知实体列表：
- 技能：{all_skills}
- 城市：{all_cities}
- 公司：{all_companies}
- 岗位：{all_jobs}

请判断用户意图并抽取槽位，严格返回 JSON：
{{
  "intent": "recommend | salary | job_skills | company_jobs | city_jobs | unknown",
  "skills": [], "cities": [], "companies": [], "jobs": []
}}

意图说明：
- recommend：用户想找推荐工作，需抽取 skills
- salary：用户问薪资，需抽取 jobs 或 cities
- job_skills：用户问某岗位需要什么技能，需抽取 jobs
- company_jobs：用户问某公司招什么岗位，需抽取 companies
- city_jobs：用户问某城市有哪些岗位，需抽取 cities
- unknown：无法识别

注意：只能从上面已知实体列表中抽取，不要编造。"""
        raw = self._chat(system, user, json_mode=True)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def polish(self, query, structured_answer):
        system = ("你是一个求职助手，请把结构化查询结果转成自然、友好的中文回答，"
                  "不要编造信息，不要加不存在的岗位。")
        user = f"""用户问：{query}

知识图谱查询结果：
{structured_answer}

请用自然语言复述，保留所有岗位名、公司、城市、薪资、技能。"""
        out = self._chat(system, user)
        return out if out else structured_answer


# ============================================================
# 问答系统
# ============================================================
class JobKGQA:
    def __init__(self, G):
        self.G = G
        self.llm = LLM(client) if USE_LLM else None
        self.all_skills = [n.split(":", 1)[1] for n, d in G.nodes(data=True) if d["type"] == "skill"]
        self.all_cities = [n.split(":", 1)[1] for n, d in G.nodes(data=True) if d["type"] == "city"]
        self.all_companies = [n.split(":", 1)[1] for n, d in G.nodes(data=True) if d["type"] == "company"]
        self.all_jobs = [n.split(":", 1)[1] for n, d in G.nodes(data=True) if d["type"] == "job"]

    def _out(self, node, rel):
        return [v for _, v, d in self.G.out_edges(node, data=True) if d["rel"] == rel]

    def _in(self, node, rel):
        return [u for u, _, d in self.G.in_edges(node, data=True) if d["rel"] == rel]

    def rule_parse(self, q):
        found_skills = [s for s in self.all_skills if s.lower() in q.lower()]
        found_cities = [c for c in self.all_cities if c in q]
        found_companies = [c for c in self.all_companies if c in q]
        found_jobs = [j for j in self.all_jobs if j in q]

        if any(k in q for k in ["推荐", "适合", "能做什么", "找什么工作", "匹配"]):
            return {"intent": "recommend", "skills": found_skills,
                    "cities": [], "companies": [], "jobs": []}
        if any(k in q for k in ["薪资", "工资", "多少钱", "待遇", "薪水"]):
            return {"intent": "salary", "skills": [], "cities": found_cities,
                    "companies": [], "jobs": found_jobs}
        if any(k in q for k in ["需要什么技能", "要求", "要会什么", "技能要求"]):
            return {"intent": "job_skills", "skills": [], "cities": [],
                    "companies": [], "jobs": found_jobs}
        if any(k in q for k in ["招什么", "招聘", "有哪些岗位", "在招"]):
            return {"intent": "company_jobs", "skills": [], "cities": found_cities,
                    "companies": found_companies, "jobs": []}
        if found_cities:
            return {"intent": "city_jobs", "skills": [], "cities": found_cities,
                    "companies": [], "jobs": []}
        return {"intent": "unknown", "skills": [], "cities": [], "companies": [], "jobs": []}

    def parse(self, q):
        if self.llm:
            result = self.llm.parse_intent(
                q, self.all_skills, self.all_cities, self.all_companies, self.all_jobs)
            if result and result.get("intent"):
                return result
        return self.rule_parse(q)

    def recommend(self, skills, cities=None, top_k=5):
        user_set = set(skills)
        if not user_set:
            return "请告诉我你会哪些技能，例如：我会 Python、SQL、MySQL。"
        results = []
        for job_node in [n for n, d in self.G.nodes(data=True) if d["type"] == "job"]:
            required = [v.split(":", 1)[1] for v in self._out(job_node, "requires")]
            if not required:
                continue
            if cities:
                job_cities = [x.split(":", 1)[1] for x in self._out(job_node, "located_in")]
                if not set(cities) & set(job_cities):
                    continue
            hit = user_set & set(required)
            miss = set(required) - user_set
            score = 0.5 * (len(hit) / len(user_set)) + 0.5 * (len(hit) / len(required))
            results.append({
                "job": job_node.split(":", 1)[1],
                "score": round(score, 3),
                "hit": sorted(hit), "miss": sorted(miss),
                "salary": (self.G.nodes[job_node].get("salary_min"),
                           self.G.nodes[job_node].get("salary_max")),
                "city": [x.split(":", 1)[1] for x in self._out(job_node, "located_in")],
                "company": [x.split(":", 1)[1] for x in self._out(job_node, "offered_by")],
            })
        results.sort(key=lambda x: (-x["score"], -len(x["hit"])))

        lines = [f"根据你会 [{', '.join(sorted(user_set))}]，推荐如下：\n"]
        for i, r in enumerate(results[:top_k], 1):
            lines.append(
                f"{i}. {r['job']}（匹配度 {r['score'] * 100:.0f}%）\n"
                f"   公司：{', '.join(r['company'])}  城市：{', '.join(r['city'])}\n"
                f"   薪资：{r['salary'][0]}-{r['salary'][1]}K\n"
                f"   已匹配：{', '.join(r['hit']) or '无'}\n"
                f"   还需补充：{', '.join(r['miss']) or '无'}\n"
            )
        return "\n".join(lines)

    def query_salary(self, jobs, cities):
        rows = []
        for j in jobs:
            node = f"job:{j}"
            if not self.G.has_node(node):
                continue
            mn = self.G.nodes[node].get("salary_min")
            mx = self.G.nodes[node].get("salary_max")
            if cities:
                jc = [x.split(":", 1)[1] for x in self._out(node, "located_in")]
                if not set(cities) & set(jc):
                    continue
            rows.append(f"{j}：{mn}-{mx}K")
        return "\n".join(rows) if rows else "没有找到对应的薪资信息。"

    def query_job_skills(self, jobs):
        rows = []
        for j in jobs:
            node = f"job:{j}"
            if not self.G.has_node(node):
                continue
            skills = [v.split(":", 1)[1] for v in self._out(node, "requires")]
            edu = [v.split(":", 1)[1] for v in self._out(node, "requires_edu")]
            exp = [v.split(":", 1)[1] for v in self._out(node, "requires_exp")]
            rows.append(
                f"{j}\n  技能：{', '.join(skills)}\n"
                f"  学历：{', '.join(edu)}  经验：{', '.join(exp)}")
        return "\n\n".join(rows) if rows else "没有找到该岗位。"

    def query_company_jobs(self, companies, cities):
        rows = []
        for c in companies:
            node = f"company:{c}"
            if not self.G.has_node(node):
                continue
            jobs = [u.split(":", 1)[1] for u in self._in(node, "offered_by")]
            if cities:
                jobs = [j for j in jobs
                        if set(cities) & set(x.split(":", 1)[1]
                                             for x in self._out(f"job:{j}", "located_in"))]
            rows.append(f"{c} 在招岗位：{', '.join(jobs) or '无'}")
        return "\n".join(rows) if rows else "没有找到该公司。"

    def query_city_jobs(self, cities):
        rows = []
        for c in cities:
            node = f"city:{c}"
            if not self.G.has_node(node):
                continue
            jobs = [u.split(":", 1)[1] for u in self._in(node, "located_in")]
            rows.append(f"{c} 的岗位：{', '.join(jobs) or '无'}")
        return "\n".join(rows) if rows else "没有找到该城市。"

    def answer(self, q):
        slots = self.parse(q)
        intent = slots.get("intent", "unknown")

        if intent == "recommend":
            raw = self.recommend(slots.get("skills", []), slots.get("cities", []))
        elif intent == "salary":
            raw = self.query_salary(slots.get("jobs", []), slots.get("cities", []))
        elif intent == "job_skills":
            raw = self.query_job_skills(slots.get("jobs", []))
        elif intent == "company_jobs":
            raw = self.query_company_jobs(slots.get("companies", []), slots.get("cities", []))
        elif intent == "city_jobs":
            raw = self.query_city_jobs(slots.get("cities", []))
        else:
            raw = ("没听懂。你可以这样问：\n"
                   "  - 我会 Python 和 SQL，推荐什么工作？\n"
                   "  - 算法工程师需要什么技能？\n"
                   "  - 字节跳动招什么岗位？\n"
                   "  - Python后端工程师薪资多少？\n"
                   "  - 北京有哪些岗位？")

        if self.llm and intent != "unknown":
            polished = self.llm.polish(q, raw)
            return {"answer": polished, "raw": raw, "intent": intent, "slots": slots}
        return {"answer": raw, "raw": raw, "intent": intent, "slots": slots}
# -*-coding:utf-8-*-
