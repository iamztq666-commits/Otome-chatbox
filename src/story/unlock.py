import os, glob
import chromadb
from sentence_transformers import SentenceTransformer
from src.rag.plot_parser import load_plot_docs

CHROMA_DIR = "./chroma"
CANON_DIR = "./data/canon"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk(text: str, size=800, overlap=120):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection("canon")

    emb = SentenceTransformer(EMB_MODEL)

    # 清空重建（开发期方便）
    try:
        client.delete_collection("canon")
    except Exception:
        pass
    col = client.get_or_create_collection("canon")

    ids, docs, metas = [], [], []
    idx = 0

    # 1) persona/world/timeline：默认 stage_min=0, spoiler=0
    for fp in glob.glob(os.path.join(CANON_DIR, "*.md")):
        name = os.path.basename(fp)
        if name == "plot.md":
            continue
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()

        doc_type = name.replace(".md", "")
        for ch in chunk(text):
            ids.append(f"{doc_type}-{idx}")
            docs.append(ch)
            metas.append({
                "type": doc_type,
                "stage_min": 0,
                "trust_min": 0,
                "spoiler_level": 0,
                "source": name
            })
            idx += 1

    # 2) plot：每个节点独立 metadata（stage_min/trust_min/spoiler）
    plot_path = os.path.join(CANON_DIR, "plot.md")
    plot_docs = load_plot_docs(plot_path)

    for p in plot_docs:
        # plot 不用切太碎，避免丢 meta；这里轻切分
        for j, ch in enumerate(chunk(p.text, size=900, overlap=150)):
            ids.append(f"plot-{p.plot_id}-{j}")
            docs.append(ch)
            metas.append({
                "type": "plot",
                "plot_id": p.plot_id,
                "title": p.title,
                "stage_min": p.stage_min,
                "trust_min": p.trust_min,
                "spoiler_level": p.spoiler_level,
                "keywords": ",".join(p.keywords),
                "source": "plot.md",
            })

    vecs = emb.encode(docs, show_progress_bar=True).tolist()
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)

    print(f"Ingested: {len(ids)} chunks into {CHROMA_DIR}/canon")

if __name__ == "__main__":
    main()
