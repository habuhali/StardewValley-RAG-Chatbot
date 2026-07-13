import os
import streamlit as st
import dashscope
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

# 加载密钥
env_path = find_dotenv()
load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8")
dash_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = dash_key

# 1. 向量模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=dash_key
)

# 2. 加载本地知识库
def load_retriever():
    loader = DirectoryLoader(
        path="./stardew_data",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embedding)
    return db.as_retriever(search_kwargs={"k": 3})

# 3. 原生通义千问问答函数（修复检索方法）
def get_answer(retriever, user_q):
    # 修复：新版用invoke()替代get_relevant_documents
    context_docs = retriever.invoke(user_q)
    context_text = "\n".join([doc.page_content for doc in context_docs])
    # 拼接完整提示词
    prompt = f"""
你是一名专业的【星露谷物语AI智能客服】，只解答星露谷物语游戏相关问题。
严格遵守以下规则：
1. 你的所有回答必须仅基于下面提供的参考知识库内容，绝对禁止编造游戏数据；
2. 若参考资料里没有对应答案，直接回复：“抱歉，暂无该星露谷相关资料，无法解答”；
3. 回答条理清晰，分类明确（作物、NPC、钓鱼、挖矿、献祭、赚钱攻略）；
4. 非星露谷问题，统一礼貌拒绝回答。

参考资料：
{context_text}

用户问题：{user_q}
"""
    # 调用通义千问原生接口
    resp = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        result_format="message"
    )
    return resp.output.choices[0].message.content

# Streamlit网页界面
st.set_page_config(page_title="星露谷物语 AI 智能客服", page_icon="🌾")
st.title("🌾 星露谷物语 AI 智能客服")
st.caption("基于本地RAG知识库，仅解答星露谷物语游戏相关问题")

# 缓存检索器，只加载一次知识库
if "retriever" not in st.session_state:
    with st.spinner("正在加载星露谷游戏知识库..."):
        st.session_state.retriever = load_retriever()
    st.success("知识库加载完成，可以提问！")

# 用户输入
user_input = st.text_input("输入你的星露谷问题：")
if user_input:
    with st.spinner("AI正在检索资料并生成回答..."):
        reply = get_answer(st.session_state.retriever, user_input)
    st.markdown(" AI客服回答")
    st.write(reply)
