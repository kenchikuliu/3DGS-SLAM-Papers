#!/usr/bin/env python3
"""
RAG向量数据库构建脚本
将extracted_markdown/目录中的Markdown文件切分、嵌入，构建本地ChromaDB向量索引
依赖: pip install chromadb sentence-transformers langchain
"""

import os
import json
import time
from pathlib import Path
from typing import Iterator

# ── 配置 ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
MD_DIR = BASE_DIR / "extracted_markdown"
DB_DIR = BASE_DIR / "chroma_db"

# 嵌入模型（本地运行，无需API）
EMBED_MODEL = "BAAI/bge-m3"   # 支持中英双语，适合学术文章
# 替代选项（如GPU内存不足）: "BAAI/bge-small-en-v1.5" / "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000      # 每个chunk的字符数
CHUNK_OVERLAP = 200    # chunk之间的重叠
COLLECTION_NAME = "3dgs_slam_papers"
BATCH_SIZE = 64        # 嵌入批次大小

def chunk_markdown(text: str, file_stem: str) -> list[dict]:
    """将Markdown文本按段落切分为chunks"""
    # 按双换行分段
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_len = 0
    section_title = ""

    for para in paragraphs:
        # 记录当前章节标题
        if para.startswith('#'):
            section_title = para.lstrip('#').strip()

        para_len = len(para)

        if current_len + para_len > CHUNK_SIZE and current_chunk:
            # 当前chunk已满，保存并开始新chunk（保留overlap）
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "source": file_stem,
                "section": section_title,
                "chunk_idx": len(chunks)
            })
            # 保留最后一段作为overlap
            overlap_paras = current_chunk[-1:] if current_len > CHUNK_OVERLAP else current_chunk
            current_chunk = overlap_paras.copy()
            current_len = sum(len(p) for p in current_chunk)

        current_chunk.append(para)
        current_len += para_len

    # 最后一个chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "source": file_stem,
            "section": section_title,
            "chunk_idx": len(chunks)
        })

    return chunks

def build_rag_database():
    """构建向量数据库"""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("请先安装依赖: pip install chromadb sentence-transformers")
        return

    md_files = sorted(MD_DIR.glob("*.md"))
    print(f"找到 {len(md_files)} 个Markdown文件")

    if not md_files:
        print(f"错误: {MD_DIR} 目录为空，请先运行 mineru_batch_processor.py")
        return

    # 初始化ChromaDB
    DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # 删除已有collection（重建）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"已删除旧collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 加载嵌入模型
    print(f"\n加载嵌入模型: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL, device="cuda" if _has_cuda() else "cpu")
    print("模型加载完成")

    # 处理所有文件
    all_chunks = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(text, md_file.stem)
        all_chunks.extend(chunks)

    print(f"\n共 {len(all_chunks)} 个chunks，开始嵌入...")

    # 批量嵌入并写入数据库
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        ids = [f"{c['source']}__chunk{c['chunk_idx']}" for c in batch]
        metadatas = [{"source": c["source"], "section": c["section"], "chunk_idx": c["chunk_idx"]} for c in batch]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (i // BATCH_SIZE + 1) % 10 == 0 or i + BATCH_SIZE >= len(all_chunks):
            print(f"  进度: {min(i+BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks")

    print(f"\n数据库构建完成！位置: {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}，共 {collection.count()} 条记录")

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def search_example():
    """演示搜索功能"""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL, device="cuda" if _has_cuda() else "cpu")

    queries = [
        "3D Gaussian Splatting for SLAM tracking",
        "loop closure detection in Gaussian SLAM",
        "dynamic scene reconstruction with Gaussians",
    ]

    print("\n=== 搜索演示 ===")
    for query in queries:
        print(f"\nQuery: {query}")
        embedding = model.encode([query], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        for j, (doc, meta, dist) in enumerate(zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )):
            print(f"  [{j+1}] source={meta['source'][:40]} score={1-dist:.3f}")
            print(f"       {doc[:150]}...")

if __name__ == "__main__":
    build_rag_database()
    # 构建完成后演示搜索
    # search_example()
