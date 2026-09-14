# 知识库机器人（Knowledge Bot）

一个从零手写的 LLM 应用项目，实现了 RAG 检索增强生成 + Agent 自主工具调用 + 流式输出 + 多轮对话的完整知识库问答系统。

## 功能特性

- **RAG 混合检索**：向量检索 + BM25 关键词检索 + Reranker 重排序，三路召回
- **Agent 自主决策**：大模型自主判断调用哪个工具（知识库检索 / 计算器 / 时间 / 天气）
- **流式输出**：SSE 逐字渲染，带思考过程和工具调用标签
- **多轮对话**：session_id 会话管理，历史自动摘要压缩，支持 Redis / 内存双模式
- **Query Rewrite**：多轮对话中自动把带代词的问题改写成独立查询
- **模型路由**：根据问题类型自动选择推理模型
- **结构化分块**：Markdown 按标题层级切分，支持 .md / .txt / .docx / .pdf 四种格式
- **工程化**：日志、评估、缓存、配置安全、多环境（dev/test/prod）、容错重试
- **用户反馈**：好评问答对自动收录进知识库，差评记录待 review

## 技术栈

| 层 | 技术 |
|---|---|
| 大模型 | DeepSeek Chat（回答生成） |
| Embedding | 硅基流动 BGE-large-zh-v1.5 |
| 重排序 | 硅基流动 BGE Reranker v2-m3 |
| 向量库 | Chroma（持久化） |
| 关键词检索 | rank_bm25 |
| Web 框架 | FastAPI + uvicorn |
| 前端 | 原生 HTML + marked.js（Markdown 渲染） |
| 会话存储 | Redis（可选，默认内存回退） |
| 文档解析 | python-docx / pypdf（按需导入） |

## 项目结构

```
knowledge_bot/
├── config.py          # 配置管理（从 .env 读取，不硬编码密钥）
├── common.py          # 大模型调用（chat / chat_with_tools / chat_stream / embed），重试+缓存
├── retriever.py       # 混合检索器（向量 + BM25 + 重排序）
├── ingest.py          # 建库脚本（文档解析 → 结构化分块 → 向量化 → 入库）
├── agent.py           # Agent 核心（工具定义 / 实现 / ReAct 循环 / Query Rewrite / 历史压缩）
├── api.py             # FastAPI 后端（/ask /ask/stream /history /clear /feedback /reindex）
├── session_store.py   # 会话存储（Redis 优先，内存回退）
├── ask.py             # CLI 命令行问答
├── dev.py             # 统一启动脚本（dev/test/prod 三环境）
├── evaluate.py        # RAG 效果评估脚本
├── ab_test.py         # Prompt A/B 测试脚本
├── inspect_db.py      # 查看向量库内容
├── index.html         # 聊天前端页面
├── requirements.txt   # Python 依赖
├── .env.example       # 环境变量模板（复制为 .env 后填密钥）
├── .gitignore
└── docs/              # 私有知识库文档（不上传 Git，需自行准备）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```
APP_ENV=dev
LLM_API_KEY=你的DeepSeek_API_Key
EMBED_API_KEY=你的硅基流动_API_Key
# REDIS_URL=redis://localhost:6379/0   # 可选，不填则用内存存储
```

- DeepSeek API Key：https://platform.deepseek.com
- 硅基流动 API Key：https://cloud.siliconflow.cn（免费额度够用）

### 3. 准备知识库文档

在 `docs/` 目录下放入你的文档，支持格式：
- Markdown（.md）—— 按标题结构化分块
- 纯文本（.txt）
- Word（.docx）
- PDF（.pdf）

### 4. 建库

```bash
python ingest.py
```

运行后会在 `chroma_db/` 目录生成持久化向量库。新增文档后重新运行即可增量更新。

### 5. 启动服务

```bash
python dev.py           # dev 环境（127.0.0.1:8000，自动重载，自动开浏览器）
python dev.py test      # test 环境
python dev.py prod      # prod 环境
```

启动后浏览器自动打开 `http://127.0.0.1:8000`，即可开始问答。

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| GET | `/health` | 健康检查 |
| GET | `/history?session_id=xxx` | 获取会话历史 |
| POST | `/ask` | 非流式问答 |
| POST | `/ask/stream` | 流式问答（SSE） |
| POST | `/clear` | 清空会话 |
| POST | `/feedback` | 提交用户反馈（好评收录知识库，差评待 review） |
| POST | `/reindex` | 重建知识库索引 |

### 流式问答示例

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "session_id": "test-001"}'
```

SSE 事件类型：
- `thought`：模型思考过程
- `tool`：工具调用（工具名 + 参数 + 结果）
- `token`：回答逐字输出
- `done`：结束，返回更新后的对话历史

## Agent 可用工具

| 工具 | 功能 |
|---|---|
| `search_knowledge_base` | 检索私有知识库（混合检索 + 重排序） |
| `calculator` | 数学计算（强制调用，禁止心算） |
| `get_current_time` | 获取当前时间 |
| `get_weather` | 获取城市天气（wttr.in 免费 API） |
| `reindex` | 重建知识库索引（需人工确认） |

## 常用命令

```bash
# 建库 / 更新库
python ingest.py

# 启动服务
python dev.py

# CLI 问答（不开网页）
python ask.py "你的问题"

# 查看向量库内容
python inspect_db.py

# 运行评估（检索命中率 + 回答质量）
python evaluate.py

# Prompt A/B 测试
python ab_test.py
```

## 配置说明

关键配置项在 `config.py` 中，可通过环境变量覆盖：

| 配置 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | 500 | 文档分块大小（字符） |
| `CHUNK_OVERLAP` | 50 | 分块重叠大小 |
| `TOP_K` | 3 | 检索返回片段数 |
| `MAX_TOOL_RESULT_LENGTH` | 3000 | 单个工具返回结果上限 |
| `MIN_RERANK_SCORE` | 0.3 | 重排序相关度阈值 |
| `MAX_HISTORY_MESSAGES` | 20 | 单会话最大历史消息数 |
| `SESSION_EXPIRE_SECONDS` | 86400 | 会话过期时间（秒） |

## 设计要点

- **全异步**：所有 IO 操作使用 async/await + httpx，支持高并发
- **零静默错误**：项目中无 `pass` 吞异常，所有错误有日志或友好提示
- **缓存**：Embedding 结果持久化缓存，重复文档不重复调用 API
- **容错**：指数退避重试、工具结果截断、异常转友好提示
- **增量更新**：建库使用 upsert，不删旧数据，新增文档自动追加

## License

MIT
