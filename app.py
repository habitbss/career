# app.py
"""
Streamlit Web 界面（纯 Python）
运行：streamlit run app.py
"""
import pickle
from pathlib import Path
import streamlit as st
from llm_qa import JobKGQA

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="就业推荐问答系统",
    page_icon="💼",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"


# ---------- 加载知识图谱（只加载一次） ----------
@st.cache_resource
def load_qa():
    with open(DATA_DIR / "kg.pkl", "rb") as f:
        G = pickle.load(f)
    return JobKGQA(G)


qa = load_qa()


# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("💼 就业推荐问答")
    st.caption("基于知识图谱 + DeepSeek")

    if qa.llm:
        st.success("✅ DeepSeek 已启用")
    else:
        st.warning("⚠️ 规则模式（未配置 API Key）")

    st.divider()
    st.subheader("📊 图谱统计")
    col1, col2 = st.columns(2)
    col1.metric("节点数", qa.G.number_of_nodes())
    col2.metric("边数", qa.G.number_of_edges())
    col1.metric("岗位", len(qa.all_jobs))
    col2.metric("公司", len(qa.all_companies))
    col1.metric("技能", len(qa.all_skills))
    col2.metric("城市", len(qa.all_cities))

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()


# ---------- 主界面 ----------
st.title("💼 就业推荐问答系统")
st.caption("输入你的技能、城市或目标岗位，我来帮你推荐合适的工作")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "你好！我会根据你的技能推荐合适的工作。\n\n"
                    "试着问我：\n"
                    "- 我会 Python、SQL、MySQL，推荐什么工作？\n"
                    "- 算法工程师需要什么技能？\n"
                    "- 字节跳动招什么岗位？\n"
                    "- Python后端工程师薪资多少？\n"
                    "- 北京有哪些岗位？"}
    ]
if "pending" not in st.session_state:
    st.session_state.pending = None


# ---------- 展示历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- 示例按钮 ----------
if len(st.session_state.messages) <= 1:
    st.markdown("##### 🎯 试试这些问题：")
    samples = [
        "我会 Python、SQL、MySQL，推荐什么工作？",
        "算法工程师需要什么技能？",
        "字节跳动招什么岗位？",
        "Python后端工程师薪资多少？",
        "北京有哪些岗位？",
        "我想在杭州做大数据开发，需要会什么？",
    ]
    cols = st.columns(3)
    for i, s in enumerate(samples):
        if cols[i % 3].button(s, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()


# ---------- 处理示例按钮点击 ----------
prompt = None
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None


# ---------- 聊天输入框（底部） ----------
user_input = st.chat_input("输入你的问题...")
if user_input:
    prompt = user_input


# ---------- 处理提问 ----------
if prompt:
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用问答系统
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = qa.answer(prompt)
        st.markdown(result["answer"])
        with st.expander("🔍 查看意图识别详情"):
            st.json({
                "intent": result["intent"],
                "slots": result["slots"],
                "raw": result["raw"],
            })

    # 3. 保存到历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
    })