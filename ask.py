"""
问答脚本：从 Chroma 检索相关片段，调大模型生成回答（异步版本）。

运行：
    python ask.py
"""
import asyncio

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME, TOP_K
from common import embed, chat


async def ask(question):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # 检索
    q_vec = await embed(question)
    results = collection.query(query_embeddings=[q_vec], n_results=TOP_K)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    print("=== 检索到的片段 ===")
    for i, (doc, src) in enumerate(zip(docs, sources)):
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        print(f"  [{i + 1}] （来自 {src}）{preview}")
    print()

    context = "\n".join(f"[片段{i + 1}] {doc}" for i, doc in enumerate(docs))
    prompt = f"""请根据以下资料回答问题。如果资料中没有答案，请说"根据现有资料无法回答"。

资料：
{context}

问题：{question}
"""

    return await chat([
        {"role": "system", "content": "你是一个基于资料回答问题的助手，只依据提供的资料回答，不编造。"},
        {"role": "user", "content": prompt},
    ])


async def main():
    print("知识库问答机器人（输入 quit 或空行退出）\n")
    while True:
        try:
            q = input("你的问题：").strip()
        except EOFError:
            q = ""
        if q.lower() in ("quit", "exit", ""):
            break
        try:
            answer = await ask(q)
            print(f"\n回答：{answer}\n")
        except Exception as e:
            print(f"\n出错了：{e}\n")


if __name__ == "__main__":
    asyncio.run(main())
