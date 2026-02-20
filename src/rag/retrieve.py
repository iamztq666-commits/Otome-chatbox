import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection("canon")
emb = SentenceTransformer(EMB_MODEL)


def _where_filter(filter_type: str, stage: int, affection: int, memory_unlock: int):
    """
    Chroma where 过滤：
    - type 必须匹配
    - stage_min <= stage
    - affection_min <= affection
    - memory_unlock <= memory_unlock
    """
    # Chroma 支持 $and / $lte 等（不同版本略有差异，但这类写法通常可用）
    return {
        "$and": [
            {"type": filter_type},
            {"stage_min": {"$lte": int(stage)}},
            {"affection_min": {"$lte": int(affection)}},
            {"memory_unlock": {"$lte": int(memory_unlock)}},
        ]
    }


def search(query: str, filter_type: str, stage: int, affection: int, memory_unlock: int, k: int = 2):
    """
    返回 list[tuple[str, dict]] -> (doc, meta)
    """
    q = (query or "").strip()
    if not q:
        return []

    qvec = emb.encode([q]).tolist()[0]
    where = _where_filter(filter_type, stage, affection, memory_unlock)

    try:
        res = col.query(
            query_embeddings=[qvec],
            n_results=int(k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # 兜底：如果 where 语法/版本不兼容，则不做数值门槛过滤，仅按 type 过滤
        res = col.query(
            query_embeddings=[qvec],
            n_results=int(k),
            where={"type": filter_type},
            include=["documents", "metadatas", "distances"],
        )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    pairs = []
    for d, m in zip(docs, metas):
        if not d:
            continue
        pairs.append((d, m or {}))
    return pairs


def retrieve_for_otome(query: str, stage: int, affection: int, memory_unlock: int, k: int = 6):
    """
    乙游检索策略：
    1) persona（稳定设定）优先
    2) memory（共同回忆）其次
    3) world（世界观）少量兜底
    4) plot（剧情片段）按解锁逐步放开

    参数说明：
    - stage: 关系阶段 0~3
    - affection: 亲密度/好感度 0~100（你 state.trust）
    - memory_unlock: 解锁等级 0~2
    """
    pairs = []

    # 先拿人设与回忆
    pairs += search(query, "persona", stage, affection, memory_unlock, k=2)
    pairs += search(query, "memory", stage, affection, memory_unlock, k=2)

    # 再补世界观/剧情（按解锁放开）
    if memory_unlock >= 1:
        pairs += search(query, "world", stage, affection, memory_unlock, k=1)
    if memory_unlock >= 2:
        pairs += search(query, "plot", stage, affection, memory_unlock, k=2)

    # 去重：按 (source/title/plot_id + doc前缀) 粗去重
    seen = set()
    uniq = []
    for doc, meta in pairs:
        key = (
            meta.get("type"),
            meta.get("source") or "",
            meta.get("plot_id") or meta.get("title") or "",
            doc.strip()[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append((doc, meta))

    return uniq[:k]
