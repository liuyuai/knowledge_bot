"""共享函数：向量化（embed）和大模型调用（chat）—— 全异步版本
新增：API 重试（指数退避）、Embedding 结果缓存（持久化到 embed_cache.json）
"""
import asyncio
import hashlib
import json
import os

import httpx

from config import (
    EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
)


# ==================== 异步重试工具 ====================
async def _retry_async(func, max_retries=3, base_delay=1.0, label="API"):
    """带指数退避的异步重试。
    - 网络错误 / 超时 / 5xx：自动重试，等待 1s → 2s → 4s
    - 4xx（参数错误、鉴权失败）：不重试，直接抛出
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                raise  # 4xx 不重试
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[{label}] HTTP {e.response.status_code}，{delay:.0f}s 后重试...")
            await asyncio.sleep(delay)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[{label}] 网络错误: {e}，{delay:.0f}s 后重试...")
            await asyncio.sleep(delay)


# ==================== Embedding 缓存 ====================
_EMBED_CACHE_FILE = "embed_cache.json"
_embed_cache = {}


def _load_embed_cache():
    """启动时从文件加载缓存。"""
    global _embed_cache
    if os.path.exists(_EMBED_CACHE_FILE):
        try:
            with open(_EMBED_CACHE_FILE, "r", encoding="utf-8") as f:
                _embed_cache = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[common] 缓存文件损坏或无法读取，已重置为空缓存: {e}")
            _embed_cache = {}


def _save_embed_cache():
    """缓存写入文件（失败不影响主流程）。"""
    try:
        with open(_EMBED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_embed_cache, f)
    except IOError as e:
        print(f"[common] 缓存写入失败（不影响主流程）: {e}")


def _cache_key(text):
    """用文本的 MD5 作为缓存键。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# 模块加载时读取缓存
_load_embed_cache()


# ==================== 核心异步函数 ====================
async def embed(text):
    """把文本变成向量（调硅基流动 BGE，返回归一化后的向量）。
    带缓存：相同文本直接返回缓存结果，不重复调用 API。
    带重试：网络错误自动重试。
    """
    key = _cache_key(text)
    if key in _embed_cache:
        return _embed_cache[key]

    async def _call():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                EMBED_BASE_URL.rstrip("/") + "/embeddings",
                json={"model": EMBED_MODEL, "input": text},
                headers={"Authorization": "Bearer " + EMBED_API_KEY},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    try:
        vec = await _retry_async(_call, label="Embedding")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Embedding API 失败 HTTP {e.response.status_code}：{e.response.text[:300]}")

    # 归一化
    norm = sum(x * x for x in vec) ** 0.5
    result = [x / norm for x in vec] if norm > 0 else vec

    # 写入缓存
    _embed_cache[key] = result
    _save_embed_cache()
    return result


async def chat(messages, model=None):
    """调大模型生成回答。带重试（网络错误/5xx 自动重试）。
    model: 指定模型，不传则用默认 LLM_MODEL。"""
    use_model = model or LLM_MODEL
    async def _call():
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                LLM_BASE_URL.rstrip("/") + "/chat/completions",
                json={"model": use_model, "messages": messages},
                headers={"Authorization": "Bearer " + LLM_API_KEY},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    try:
        return await _retry_async(_call, label="LLM")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"LLM API 失败 HTTP {e.response.status_code}：{e.response.text[:300]}")


async def chat_with_tools(messages, tools, model=None):
    """调大模型，支持 function calling（工具调用）。
    返回完整的 message dict，可能包含 content 和 tool_calls。
    带重试（网络错误/5xx 自动重试）。
    model: 指定模型，不传则用默认 LLM_MODEL。"""
    use_model = model or LLM_MODEL
    async def _call():
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                LLM_BASE_URL.rstrip("/") + "/chat/completions",
                json={
                    "model": use_model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                },
                headers={"Authorization": "Bearer " + LLM_API_KEY},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]

    try:
        return await _retry_async(_call, label="LLM")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"LLM API 失败 HTTP {e.response.status_code}：{e.response.text[:300]}")


async def chat_stream(messages, model=None):
    """流式调用大模型，逐 token yield 文本片段（异步生成器）。
    用于前端逐字显示回答效果。
    model: 指定模型，不传则用默认 LLM_MODEL。"""
    use_model = model or LLM_MODEL
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            LLM_BASE_URL.rstrip("/") + "/chat/completions",
            json={"model": use_model, "messages": messages, "stream": True},
            headers={"Authorization": "Bearer " + LLM_API_KEY},
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
