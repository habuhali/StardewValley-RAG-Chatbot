 星露谷物语本地RAG问答客服
StardewValley Local RAG Chatbot | 轻量化私有知识库检索问答系统

一、项目介绍
基于Python + Streamlit + LangChain + FAISS + 阿里云通义千问 实现RAG检索增强问答
无需本地部署大模型，调用阿里云免费API，加载本地txt游戏攻略知识库，精准检索资料后AI作答，严格限制模型不编造内容。
项目支持本地运行 + Streamlit云端公网部署，分享链接即可全网访问AI客服页面。

1.在线演示地址（直接点开体验）
https://stardewvalley-rag-chatbot-84zoxsd8umryhdhzu43qxa.streamlit.app

二、技术栈
- 前端可视化：Streamlit
- RAG框架：LangChain Community
- 向量数据库：FAISS-CPU
- LLM & Embedding：阿里DashScope 通义千问 qwen-turbo / text-embedding-v1
- 环境管理：python-dotenv
- 云端部署：Streamlit Community Cloud（免费公网托管）

三、项目亮点
1. 纯本地私有知识库，仅读取项目内txt文档，数据不上传第三方
2. 中文文档兼容，自动适配UTF-8编码，解决Windows记事本乱码问题
3. 规避LangChain版本兼容冲突，原生封装DashScope接口，稳定无导入报错
4. 一键启动网页服务，无需前端开发基础；支持云端免费部署，生成永久公网链接
5. 环境变量隔离API密钥，本地.env、云端Secrets双方案，敏感信息不会泄露到代码仓库
6. 相似度Top3检索，限定AI仅使用参考资料回答，消除模型幻觉

四、仓库文件说明
├── app.py               # 项目主运行程序
├── requirements.txt     # 依赖包清单
├── .gitignore           # 屏蔽密钥、缓存文件
└── stardew_data/        # 本地知识库目录，存放 txt 攻略文档

五、本地启动方式
1. 安装依赖
```bash
pip install -r requirements.txt
