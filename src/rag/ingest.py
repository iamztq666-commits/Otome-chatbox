import os, glob
import chromadb
from sentence_transformers import SentenceTransformer
from src.rag.plot_parser import load_plot_docs

CHROMA_DIR = "./chroma"
CANON_DIR = "./data/canon"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk(text: str, size=459, overlap=80):
    chunks = []
    i = 0
    text = text or ""
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 清空重建（开发期方便）
    try:
        client.delete_collection("canon")
    except Exception:
        pass
    col = client.get_or_create_collection("canon")

    emb = SentenceTransformer(EMB_MODEL)

    ids, docs, metas = [], [], []
    idx = 0

    # ========= 1) data/canon/*.md（除了 plot.md） =========
    # 兼容你现在：persona.md / world.md / timeline.md ...
    for fp in glob.glob(os.path.join(CANON_DIR, "*.md")):
        name = os.path.basename(fp)
        if name == "plot.md":
            continue

        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()

        doc_type = name.replace(".md", "")  # persona/world/timeline...
        for ch in chunk(text):
            ids.append(f"{doc_type}-{idx}")
            docs.append(ch)
            metas.append({
                "type": doc_type,
                "stage_min": 0,
                "affection_min": 0,
                "memory_unlock": 0,
                "source": name,
                "title": doc_type,
            })
            idx += 1

    # ========= 2) 可选：data/canon/memory/*.md（乙游回忆库） =========
    mem_dir = os.path.join(CANON_DIR, "memory")
    if os.path.isdir(mem_dir):
        for fp in glob.glob(os.path.join(mem_dir, "*.md")):
            name = os.path.basename(fp)
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()

            for j, ch in enumerate(chunk(text)):
                ids.append(f"memory-{name}-{j}-{idx}")
                docs.append(ch)
                metas.append({
                    "type": "memory",
                    "stage_min": 1,
                    "affection_min": 30,
                    "memory_unlock": 1,
                    "source": f"memory/{name}",
                    "title": name.replace(".md", ""),
                })
                idx += 1

    # ========= 3) plot.md（仍用你现有 plot_parser 解析） =========
    plot_path = os.path.join(CANON_DIR, "plot.md")
    if os.path.exists(plot_path):
        plot_docs = load_plot_docs(plot_path)

        for p in plot_docs:
            stage_min = int(getattr(p, "stage_min", 0) or 0)
            # 兼容旧 parser 字段：trust_min / spoiler_level
            affection_min = int(getattr(p, "affection_min", None) or getattr(p, "trust_min", 0) or 0)
            memory_unlock = int(getattr(p, "memory_unlock", None) or getattr(p, "spoiler_level", 0) or 0)

            plot_id = getattr(p, "plot_id", None) or "plot"
            title = getattr(p, "title", None) or plot_id
            keywords = getattr(p, "keywords", []) or []
            if isinstance(keywords, str):
                keywords = [keywords]

            for j, ch in enumerate(chunk(getattr(p, "text", ""), size=900, overlap=150)):
                ids.append(f"plot-{plot_id}-{j}")
                docs.append(ch)
                metas.append({
                    "type": "plot",
                    "plot_id": plot_id,
                    "title": title,
                    "stage_min": stage_min,
                    "affection_min": affection_min,
                    "memory_unlock": memory_unlock,
                    "keywords": ",".join(keywords),
                    "source": "plot.md",
                })

    if not docs:
        print("No docs found. Put .md under ./data/canon (optional ./data/canon/memory) and/or plot.md")
        return

    vecs = emb.encode(docs, show_progress_bar=True).tolist()
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)

    print(f"Ingested: {len(ids)} chunks into {CHROMA_DIR}/canon")

if __name__ == "__main__":
    main()
