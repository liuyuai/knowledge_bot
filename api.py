"""
知识库机器人 - FastAPI 后端（全异步版本）

启动：
    uvicorn api:app --reload --port 8000

接口：
    GET  /health    健康检查
    POST /ask       提问，返回回答和引用来源
    POST /ask/stream 流式提问（SSE）
    POST /clear     清空会话历史
    POST /reindex   重新建库
"""
import json
import logging
import time

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import run_agent, run_agent_stream, compress_history
from config import APP_ENV, ENV_CONFIG, CHROMA_PATH, COLLECTION_NAME, TOP_K, DOCS_DIR
from common import embed
import session_store

app = FastAPI(title="知识库机器人 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 日志
_log_level = getattr(logging, ENV_CONFIG[APP_ENV]["log_level"].upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"[启动] 会话存储模式: {session_store.storage_mode()}")


def _friendly_error(e):
    """把原始异常转成用户友好的提示。"""
    msg = str(e)
    if any(k in msg for k in ["10054", "Connection reset", "远程主机强迫关闭", "连接重置", "broken pipe"]):
        return "连接中断，请重试"
    if any(k in msg.lower() for k in ["timeout", "timed out", "超时"]):
        return "请求超时，请稍后重试"
    if "401" in msg or "auth" in msg.lower():
        return "API Key 无效，请检查配置"
    if "429" in msg or "rate" in msg.lower():
        return "请求过于频繁，请稍后再试"
    return "服务暂时不可用，请稍后重试"


# ==================== 会话存储（Redis / 内存回退）====================
MAX_HISTORY_MESSAGES = 20


def _get_history(session_id):
    """读取会话历史，超过上限时截断。"""
    if not session_id:
        return None
    history = session_store.get_history(session_id)
    if history and len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
        session_store.save_history(session_id, history)
    return history


async def _save_history(session_id, history):
    """保存会话历史，超过阈值时自动压缩摘要（异步）。"""
    if session_id:
        history = await compress_history(history)
        session_store.save_history(session_id, history)


class QuestionRequest(BaseModel):
    question: str
    session_id: str = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    steps: list[str] = []


@app.get("/")
async def index():
    """首页：直接返回聊天页面。"""
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """Agent 自主决策（异步）。"""
    start = time.time()
    logger.info(f"收到问题: {req.question}")
    try:
        answer, messages, steps = await run_agent(req.question)

        sources = []
        for msg in messages:
            if msg.get("role") == "tool":
                try:
                    results = json.loads(msg["content"])
                    if isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict) and r.get("source") and r["source"] not in sources:
                                sources.append(r["source"])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"工具结果解析失败，无法提取来源: {e}")

        elapsed = time.time() - start
        logger.info(f"Agent 回答完成，耗时 {elapsed:.1f}s，步骤: {steps}")
        logger.info(f"回答: {answer}")
        logger.info("-" * 60)

        return {"answer": answer, "sources": sources, "steps": steps}

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"处理失败，耗时 {elapsed:.1f}s，错误: {e}", exc_info=True)
        logger.info("-" * 60)
        raise


@app.post("/ask/stream")
async def ask_question_stream(req: QuestionRequest):
    """流式回答（异步 SSE）：逐 token 返回，支持多轮对话。"""
    start = time.time()
    history = _get_history(req.session_id)
    logger.info(f"[流式] 收到问题: {req.question} (session={req.session_id}, 历史{len(history) if history else 0}条)")

    async def event_generator():
        try:
            async for event in run_agent_stream(req.question, history=history):
                if event["type"] == "tool":
                    logger.info(f"[流式] 调用工具: {event['name']}({event['args']})")
                elif event["type"] == "done":
                    elapsed = time.time() - start
                    logger.info(f"[流式] 完成，耗时 {elapsed:.1f}s，步骤: {event.get('steps', [])}")
                    logger.info(f"[流式] 回答: {event.get('answer', '')[:200]}")
                    logger.info("-" * 60)
                    await _save_history(req.session_id, event.get("history", []))
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[流式] 失败，耗时 {elapsed:.1f}s，错误: {e}", exc_info=True)
            logger.info("-" * 60)
            yield f"event: error\ndata: {json.dumps({'error': _friendly_error(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/clear")
async def clear_session(req: QuestionRequest):
    """清空指定会话的对话历史。"""
    if req.session_id and session_store.has_session(req.session_id):
        session_store.delete_session(req.session_id)
        logger.info(f"会话已清空: {req.session_id}")
    return {"status": "ok"}


@app.get("/history")
async def get_history(session_id: str = None):
    """获取指定会话的对话历史，用于页面刷新后恢复聊天记录。"""
    if not session_id:
        return {"history": []}
    history = session_store.get_history(session_id)
    return {"history": history or []}


@app.post("/reindex")
async def reindex():
    """重新读取 docs/ 建库（异步）。"""
    import os
    from ingest import chunk_text

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_chunks = []
    all_ids = []
    all_sources = []

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith((".md", ".txt")):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}#{i}")
            all_sources.append(filename)

    if not all_chunks:
        return {"status": "empty", "message": "docs/ 目录下没有 .md 或 .txt 文件"}

    embeddings = [await embed(c) for c in all_chunks]
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in all_sources],
    )
    return {"status": "ok", "count": len(all_chunks)}
