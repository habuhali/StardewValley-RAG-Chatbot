import os
import streamlit as st
import dashscope
import json
import time
import sqlite3
from typing import List, Dict, Any
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# 向量模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=dash_key
)

# 加载本地知识库
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

# ====================== 游戏技能工具集 Skills ======================
# 1. 作物收益计算工具
def calc_crop_profit(season: str, crop_name: str, greenhouse: bool = False) -> Dict[str, Any]:
    crop_data = {
        "防风草": {"cost": 20, "days": 4, "sell": 35, "season": ["春季"]},
        "草莓": {"cost": 100, "days": 8, "sell": 120, "season": ["春季"]},
        "蓝莓": {"cost": 80, "days": 12, "sell": 75, "season": ["夏季"]},
        "甜瓜": {"cost": 80, "days": 12, "sell": 250, "season": ["夏季"]},
        "蔓越莓": {"cost": 240, "days": 23, "sell": 75, "season": ["秋季"]}
    }
    if crop_name not in crop_data:
        return {"error": f"暂无{crop_name}作物数据"}
    info = crop_data[crop_name]
    if season not in info["season"] and not greenhouse:
        return {"warn": f"{crop_name}仅{info['season'][0]}可露天种植，非该季节需温室"}
    cycle = info["days"] if not greenhouse else info["days"] * 0.6
    profit_per = info["sell"] - info["cost"]
    daily_profit = round(profit_per / cycle, 2)
    return {
        "作物": crop_name,
        "单株成本": info["cost"],
        "单株售价": info["sell"],
        "单株纯利": profit_per,
        "生长天数": cycle,
        "日均收益": daily_profit
    }

# 2. NPC好感查询工具
def get_npc_like(npc_name: str) -> Dict[str, Any]:
    npc_data = {
        "阿比盖尔": {"最爱":["紫水晶"], "喜爱":["石英"], "厌恶":["黏土"], "生日":"秋13"},
        "亚历克斯": {"最爱":["完整早餐"], "喜爱":["鸡蛋"], "厌恶":["石英"], "生日":"夏13"},
        "海莉": {"最爱":["椰子"], "喜爱":["水果"], "厌恶":["黏土"], "生日":"春14"}
    }
    if npc_name not in npc_data:
        return {"error": f"暂无{npc_name}NPC好感数据"}
    return {"NPC": npc_name, **npc_data[npc_name]}

# 3. 社区献祭查询
def get_community_bundle(bundle_name: str) -> Dict[str, Any]:
    bundle_data = {
        "春季作物包": ["防风草、青豆、土豆、郁金香"],
        "工匠包": ["松露油、枫糖浆、奶酪、布料"],
        "鱼类包": ["鲶鱼、虹鳟、鱿鱼、鲷鱼"]
    }
    if bundle_name not in bundle_data:
        return {"error": f"无{bundle_name}献祭包信息"}
    return {"献祭包": bundle_name, "所需物品": bundle_data[bundle_name]}

# 4. 钓鱼筛选工具
def get_fish(season: str, weather: str = "任意") -> List[str]:
    fish_map = {
        "春季": ["鲤鱼、鲶鱼、太阳鱼"],
        "夏季": ["金枪鱼、虹鳟、河豚"],
        "秋季": ["三文鱼、海参、鲈鱼"]
    }
    return fish_map.get(season, [f"{season}无鱼类数据"])

# 5. 维基百科在线检索工具
def search_wiki(keyword: str) -> str:
    try:
        loader = WikipediaLoader(
            query=keyword,
            lang="zh",
            load_max_docs=1,
            doc_content_chars_max=15000
        )
        wiki_docs = loader.load()
        if not wiki_docs:
            return "维基百科未检索到相关星露谷物语内容"
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        split_wiki = splitter.split_documents(wiki_docs)
        wiki_context = "\n".join([d.page_content for d in split_wiki[:3]])
        return f"【维基百科补充资料】\n{wiki_context}"
    except Exception as e:
        return f"联网检索失败，仅使用本地知识库作答，异常信息：{str(e)}"

# 工具注册列表
TOOL_LIST = [
    {
        "name": "calc_crop_profit",
        "description": "计算星露谷作物收益，输入季节、作物名、是否温室",
        "parameters": {
            "type": "object",
            "properties": {
                "season": {"type": "string", "description": "季节：春季/夏季/秋季/冬季"},
                "crop_name": {"type": "string", "description": "作物名称"},
                "greenhouse": {"type": "boolean", "default": False}
            },
            "required": ["season", "crop_name"]
        }
    },
    {
        "name": "get_npc_like",
        "description": "查询NPC喜好、厌恶物品、生日",
        "parameters": {
            "type": "object",
            "properties": {"npc_name": {"type": "string"}},
            "required": ["npc_name"]
        }
    },
    {
        "name": "get_community_bundle",
        "description": "查询社区中心献祭包所需道具",
        "parameters": {
            "type": "object",
            "properties": {"bundle_name": {"type": "string"}},
            "required": ["bundle_name"]
        }
    },
    {
        "name": "get_fish",
        "description": "根据季节查询可钓鱼类",
        "parameters": {
            "type": "object",
            "properties": {
                "season": {"type": "string"},
                "weather": {"type": "string", "default": "任意"}
            },
            "required": ["season"]
        }
    },
    {
        "name": "search_wiki",
        "description": "本地知识库没有匹配内容时，调用该工具联网检索中文维基百科星露谷资料",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "需要查询的游戏关键词"}},
            "required": ["keyword"]
        }
    }
]

# 工具执行调度器
def run_tool(tool_name: str, args: dict):
    if tool_name == "calc_crop_profit":
        return calc_crop_profit(**args)
    elif tool_name == "get_npc_like":
        return get_npc_like(**args)
    elif tool_name == "get_community_bundle":
        return get_community_bundle(**args)
    elif tool_name == "get_fish":
        return get_fish(**args)
    elif tool_name == "search_wiki":
        return search_wiki(**args)
    return {"error": "不存在该工具"}

# ====================== 对话持久化SQLite模块 ======================
def init_sqlite():
    conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_q TEXT, ai_ans TEXT, tool_info TEXT, create_time TEXT)''')
    conn.commit()
    return conn

def save_chat(conn, q, ans, tool_info=""):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    c = conn.cursor()
    c.execute("INSERT INTO chat_log(user_q,ai_ans,tool_info,create_time) VALUES (?,?,?,?)",
              (q, ans, json.dumps(tool_info, ensure_ascii=False), t))
    conn.commit()

# 全局数据库连接
db_conn = init_sqlite()

def get_answer(retriever, user_q, history: List[Dict]):
    # 前置Hook：过滤非星露谷无关提问
    non_game_words = ["抖音、王者、原神、电脑、手机、股票、动漫"]
    for w in non_game_words:
        if w in user_q:
            return "抱歉，本客服仅解答星露谷物语相关内容，其他问题无法回答。", {}

    # 1. 优先检索本地静态知识库
    context_docs = retriever.invoke(user_q)
    context_text = "\n".join([doc.page_content for doc in context_docs])
    # 修复本地资料判断逻辑：有检索结果才判定有本地资料，不会误触发维基
    has_valid_local = len(context_docs) > 0 and any(doc.page_content.strip() for doc in context_docs)
    # 仅本地完全无匹配文档才调用维基
    if not has_valid_local:
        wiki_data = search_wiki(user_q)
        context_text = context_text + "\n【维基百科补充资料】\n" + wiki_data

    # 2. 系统提示词：本地优先，缺资料再联网维基
    sys_prompt = f"""
你是星露谷物语混合RAG智能客服Agent，拥有本地知识库+5个工具能力。
可用工具列表：
{json.dumps(TOOL_LIST, ensure_ascii=False, indent=2)}
硬性执行规则：
1. 优先使用下方【本地参考资料】回答问题；本地资料完全无相关内容时，必须调用 search_wiki 联网检索维基百科；
2. 用户询问作物收益、NPC、献祭、钓鱼数据，优先调用对应计算工具输出量化结果；
3. 所有回答只能来源于本地文档 + 维基返回内容，绝对禁止编造游戏数值、道具、季节规则；
4. 若维基检索失败，直接回复「暂无相关游戏资料，无法解答」；
5. 需要调用工具时，严格输出固定JSON格式：{{"tool": "工具名", "args": {{参数键值对}}}}
本地参考资料：
{context_text}
"""
    # 修复1：格式化历史为标准role结构，限制最多6轮避免上下文超长
    formatted_history = []
    for item in history[-6:]:
        formatted_history.append({"role": "user", "content": item["q"]})
        formatted_history.append({"role": "assistant", "content": item["ans"]})

    messages = [{"role":"system","content":sys_prompt}] + formatted_history + [{"role":"user","content":user_q}]

    # 3. 第一轮LLM：兼容choices为空，双层异常捕获
    try:
        resp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=messages,
            result_format="message"
        )
        if resp.output is None:
            return "大模型接口临时异常，仅使用本地/维基预加载资料作答", {}
        # 兼容两种返回结构：有choices / 只有text
        if hasattr(resp.output, "choices") and resp.output.choices and len(resp.output.choices) > 0:
            reply = resp.output.choices[0].message.content.strip()
        else:
            reply = resp.output.text.strip()
    except Exception as api_err:
        return f"接口调用异常：{str(api_err)}，暂时无法执行工具检索", {}

    tool_result = None
    # 识别工具调用指令并执行
    if reply.startswith("{") and reply.endswith("}"):
        try:
            call_info = json.loads(reply)
            tool_name = call_info.get("tool")
            tool_args = call_info.get("args", {})
            tool_result = run_tool(tool_name, tool_args)

            # 将工具结果二次传入大模型，生成通顺回答
            final_input = f"""
工具返回数据：{json.dumps(tool_result, ensure_ascii=False, indent=2)}
结合本地知识库资料与工具数据，完整清晰回答用户问题：{user_q}
禁止编造任何游戏信息，分点排版更易阅读。
本地参考资料：{context_text}
"""
            final_msg = messages + [{"role":"user","content":final_input}]
            try:
                resp2 = dashscope.Generation.call(model="qwen-turbo", messages=final_msg, result_format="message")
                if resp2.output is None:
                    return "二次生成回答接口异常，仅展示工具原始数据", tool_result
                if hasattr(resp2.output, "choices") and resp2.output.choices and len(resp2.output.choices) > 0:
                    final_ans = resp2.output.choices[0].message.content
                else:
                    final_ans = resp2.output.text
                return final_ans, tool_result
            except Exception:
                return f"工具结果合成回答失败，工具原始数据：{json.dumps(tool_result, ensure_ascii=False)}", tool_result
        except Exception:
            # JSON解析失败，降级纯RAG回答
            return reply, None
    return reply, None

# ===================== Streamlit页面交互 =====================
# 缓存检索器，只加载一次知识库
if "retriever" not in st.session_state:
    with st.spinner("正在加载星露谷游戏知识库..."):
        st.session_state.retriever = load_retriever()
    st.success("知识库加载完成，可以提问！")

# 初始化多轮对话记忆
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏：对话历史、清空按钮
with st.sidebar:
    st.subheader("📜 对话历史记录")
    if st.button("一键清空全部对话"):
        st.session_state.chat_history = []
    # 展示最近10轮对话
    for item in st.session_state.chat_history[-10:]:
        st.markdown(f"**用户：** {item['q']}")
        st.markdown(f"**AI：** {item['ans']}")
        st.divider()

user_input = st.text_input("输入你的星露谷物语游戏问题：")
if user_input.strip():
    with st.spinner("🔍 本地知识库检索 / 工具计算 / 联网维基检索中..."):
        ans, tool_data = get_answer(
            retriever=st.session_state.retriever,
            user_q=user_input,
            history=st.session_state.chat_history
        )
    # 保存上下文记忆
    st.session_state.chat_history.append({"q": user_input, "ans": ans})
    # 存入本地日志数据库
    save_chat(db_conn, user_input, ans, tool_data)

    st.divider()
    st.subheader("🌾 AI智能客服回答")
    st.write(ans)
    # 展示工具调用明细
    if tool_data:
        st.info(f"本次执行工具返回数据：\n{json.dumps(tool_data, ensure_ascii=False, indent=2)}")