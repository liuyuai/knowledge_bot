"""
检索器：混合检索（向量 + BM25关键词）+ 重排序 —— 全异步版本
=================================================
  1. 向量检索 top 20（语义相似）
  2. BM25 关键词检索 top 20（字面匹配）
  3. 合并去重
  4. 硅基流动 rerank 模型精排，取 top 3
"""
import json

import chromadb
import httpx
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
    EMBED_API_KEY,
    EMBED_BASE_URL,
)
from common import embed

# 重排序模型（硅基流动）
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_URL = f"{EMBED_BASE_URL}/rerank"

# 混合检索粗召回数量
HYBRID_CANDIDATES = 20

# 重排序最低相关度阈值，低于此分数的结果被过滤（BGE Reranker 分数范围 0~1）
MIN_RERANK_SCORE = 0.3


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


async def _vector_search(query, n_results=HYBRID_CANDIDATES):
    """向量检索：返回 [(content, source), ...]"""
    collection = _get_collection()
    q_vec = await embed(query)
    results = collection.query(query_embeddings=[q_vec], n_results=n_results)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(docs, sources))


def _bm25_search(query, n_results=HYBRID_CANDIDATES):
    """BM25 关键词检索：返回 [(content, source), ...]（纯计算，无需异步）"""
    collection = _get_collection()
    all_docs = collection.get()
    docs = all_docs["documents"]
    sources = [m["source"] for m in all_docs["metadatas"]]

    if not docs:
        return []

    # 中文按字符级分词（简单有效；更精准可用 jieba）
    tokenized_corpus = [list(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)

    # 按分数降序，取 top n，过滤掉分数为 0 的
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in ranked[:n_results] if score > 0]

    return [(docs[i], sources[i]) for i in top_indices]


def _merge_results(*result_lists):
    """合并多路检索结果，按内容前100字符去重，保持顺序（向量优先）。"""
    seen = set()
    merged = []
    for results in result_lists:
        for content, source in results:
            key = content[:100]
            if key not in seen:
                seen.add(key)
                merged.append((content, source))
    return merged


async def _rerank(query, candidates, top_k=TOP_K):
    """
    调用硅基流动重排序 API 精排（异步）。
    返回 [(content, source, score), ...]，按相关度降序。
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return [(c, s, 1.0) for c, s in candidates]
    if not EMBED_API_KEY:
        # 没有 API Key 时跳过重排序，直接返回前 top_k
        return [(c, s, 0.0) for c, s in candidates[:top_k]]

    docs = [c for c, _ in candidates]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            RERANK_URL,
            json={"model": RERANK_MODEL, "query": query, "documents": docs},
            headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        )
        resp.raise_for_status()
        data = resp.json()

    # results: [{"index": 0, "relevance_score": 0.95}, ...]
    ranked = sorted(data["results"], key=lambda x: x["relevance_score"], reverse=True)
    # 过滤掉相关度低于阈值的结果
    filtered = [r for r in ranked if r["relevance_score"] >= MIN_RERANK_SCORE]
    # 如果过滤后为空，保留分数最高的一个（避免完全没结果）
    if not filtered and ranked:
        filtered = [ranked[0]]
    top = filtered[:top_k]

    return [
        (candidates[r["index"]][0], candidates[r["index"]][1], r["relevance_score"])
        for r in top
    ]


async def hybrid_search(query, top_k=TOP_K):
    """
    混合检索 + 重排序入口（异步）。
    返回 [{"source": ..., "content": ..., "score": ...}, ...]
    """
    # 1. 两路粗召回（向量是异步，BM25是纯计算）
    vector_results = await _vector_search(query)
    bm25_results = _bm25_search(query)

    # 2. 合并去重（向量结果优先）
    candidates = _merge_results(vector_results, bm25_results)

    # 3. 重排序精排（失败时降级为直接返回前 top_k）
    try:
        ranked = await _rerank(query, candidates, top_k=top_k)
        results = [{"source": s, "content": c, "score": score} for c, s, score in ranked]
        # 调试：打印每个结果的相关度分数
        score_info = ", ".join([f"{r['source']}={r['score']:.3f}" for r in results])
        print(f"[retriever] 检索结果（{len(results)}条，阈值{MIN_RERANK_SCORE}）: {score_info}")
    except Exception as e:
        print(f"[retriever] 重排序失败，降级为粗召回前{top_k}条: {e}")
        fallback = candidates[:top_k]
        results = [{"source": s, "content": c, "score": 0.0} for c, s in fallback]

    # 4. 截断每个片段的 content，防止 JSON 整体过长被截断导致解析失败
    for r in results:
        if len(r["content"]) > 400:
            r["content"] = r["content"][:400] + "..."

    return results
