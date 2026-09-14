"""查看 Chroma 数据库里存了什么"""
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("knowledge_base")

print(f"集合名: {col.name}")
print(f"文档总数: {col.count()}")
print()

data = col.get(include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
    print(f"[{i}] 来源: {meta['source']}")
    preview = doc[:60] + "..." if len(doc) > 60 else doc
    print(f"    内容: {preview}")
    print()
