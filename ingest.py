"""
建库脚本：读取 docs/ 下的 .md/.txt 文件，分块后向量化，存入 Chroma。

运行：
    python ingest.py

每次运行会增量更新（已存在的块会被覆盖）。
"""
import asyncio
import os
import chromadb

from config import DOCS_DIR, CHROMA_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from common import embed


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按字符数分块（用于 .txt 等无结构文本）。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_markdown(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    按 markdown 标题结构分块。
    返回 [(content, heading_path), ...]
    - 先按 #/##/### 标题切分成节
    - 每节内按空行（段落）合并，超过 chunk_size 再按字符切
    - 每个块的内容前拼上标题路径，增强检索上下文
    """
    lines = text.split('\n')
    sections = []  # [(heading_path, content_lines), ...]
    heading_stack = []  # 维护标题层级：["一级", "二级"]
    current_content = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # 遇到新标题，先保存上一节
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((' > '.join(heading_stack), content))
                current_content = []

            # 更新标题栈：同级或更高级别先弹出
            level = len(stripped) - len(stripped.lstrip('#'))
            heading_text = stripped.lstrip('#').strip()
            heading_stack = heading_stack[:level - 1] + [heading_text]
        else:
            current_content.append(line)

    # 最后一节
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((' > '.join(heading_stack), content))

    # 对每节按段落合并分块
    chunks = []
    for heading_path, content in sections:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                # 能放下，拼进当前块
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 放不下，先保存当前块
                if current_chunk:
                    chunks.append((_with_heading(current_chunk, heading_path), heading_path))
                # 段落本身超过 chunk_size，按字符切
                if len(para) > chunk_size:
                    for sc in chunk_text(para, chunk_size, overlap):
                        chunks.append((_with_heading(sc, heading_path), heading_path))
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append((_with_heading(current_chunk, heading_path), heading_path))

    return chunks


def _with_heading(content, heading_path):
    """把标题路径拼到内容前面，增强检索上下文。"""
    if heading_path:
        return f"[{heading_path}]\n{content}"
    return content


def extract_text(filepath):
    """根据文件类型提取纯文本。支持 .md/.txt/.docx/.pdf。"""
    if filepath.endswith((".md", ".txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    if filepath.endswith(".docx"):
        from docx import Document
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    if filepath.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)

    return ""


async def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_chunks = []
    all_ids = []
    all_metadatas = []

    supported = (".md", ".txt", ".docx", ".pdf")

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(supported):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        text = extract_text(filepath)
        if not text.strip():
            print(f"  {filename}: 跳过（无法提取文本或内容为空）")
            continue

        if filename.endswith(".md"):
            # markdown 文件：按标题结构分块
            results = chunk_markdown(text)
            for i, (content, heading) in enumerate(results):
                all_chunks.append(content)
                all_ids.append(f"{filename}#{i}")
                all_metadatas.append({"source": filename, "heading": heading})
            print(f"  {filename}: {len(results)} 块（结构化分块）")
        else:
            # txt/docx/pdf：按字符分块
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{filename}#{i}")
                all_metadatas.append({"source": filename, "heading": ""})
            print(f"  {filename}: {len(chunks)} 块（字符分块）")

    if not all_chunks:
        print(f"docs/ 目录下没有 .md 或 .txt 文件，请先放入文档。")
        return

    # 向量化（异步并发，比逐个调用快很多）
    print(f"\n共 {len(all_chunks)} 个文本块，开始向量化...")
    embeddings = await asyncio.gather(*[embed(chunk) for chunk in all_chunks])
    print(f"  向量化完成，共 {len(embeddings)} 个向量")

    # 存入 Chroma（upsert = 有则更新，无则插入）
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )
    print(f"\n建库完成！共 {len(all_chunks)} 个块，已存入 {CHROMA_PATH}/")


if __name__ == "__main__":
    asyncio.run(main())
