import json
import os
import sys
from typing import List, Dict, Any, Tuple

# 让脚本能 import 你的 src（按你项目结构调整）
# 如果你的脚本在 scripts/，src 在 src/，一般这样就行：
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ 把这里改成你项目真实的检索函数 import 路径
# 你之前用的是 retrieve_for_otome(...)
from src.rag.retrieve import retrieve_for_otome  # ← 如果路径不对就改这里


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def get_doc_id(meta: Dict[str, Any]) -> str:
    """优先用 plot_id，其次 title/type（按你向量库 meta 实际情况调整）"""
    return str(meta.get("plot_id") or meta.get("title") or meta.get("type") or "")


def evaluate(samples: List[Dict[str, Any]], k: int = 6) -> Dict[str, Any]:
    n = 0
    hit = 0
    mrr_sum = 0.0
    miss_cases: List[Dict[str, Any]] = []

    for s in samples:
        query = s["query"]
        gt_plot_id = str(s.get("gt_plot_id", "")).strip()
        gt_title = str(s.get("gt_title", "")).strip()

        stage = int(s.get("stage", 0))
        affection = int(s.get("affection", 20))
        memory_unlock = int(s.get("memory_unlock", 0))

        # 调你的检索：返回 canon_pairs = [(doc, meta), ...]
        canon_pairs = retrieve_for_otome(
            query=query,
            stage=stage,
            affection=affection,
            memory_unlock=memory_unlock,
            k=k
        )

        retrieved_ids = [get_doc_id(meta) for _, meta in canon_pairs]

        n += 1

        # 命中规则：优先匹配 gt_plot_id；如果你用 title 标注也可支持
        found_rank = None
        if gt_plot_id:
            for idx, rid in enumerate(retrieved_ids, start=1):
                if rid == gt_plot_id:
                    found_rank = idx
                    break
        elif gt_title:
            for idx, rid in enumerate(retrieved_ids, start=1):
                if rid == gt_title:
                    found_rank = idx
                    break

        if found_rank is not None:
            hit += 1
            mrr_sum += 1.0 / found_rank
        else:
            miss_cases.append({
                "query": query,
                "gt_plot_id": gt_plot_id,
                "gt_title": gt_title,
                "stage": stage,
                "affection": affection,
                "memory_unlock": memory_unlock,
                "topk": retrieved_ids,
            })

    recall = hit / n if n else 0.0
    mrr = mrr_sum / n if n else 0.0

    return {
        "n": n,
        "k": k,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "miss_cases": miss_cases,
    }


def main():
    eval_path = os.getenv("RAG_EVAL_PATH", "data/eval/rag_eval.jsonl")
    k = int(os.getenv("RAG_EVAL_K", "6"))

    samples = load_jsonl(eval_path)
    result = evaluate(samples, k=k)

    print(f"Eval file: {eval_path}")
    print(f"N={result['n']}  Recall@{k}={result['recall_at_k']:.3f}  MRR@{k}={result['mrr_at_k']:.3f}")

    # 可选：把 miss cases 写出来便于你调试检索
    out_path = "data/eval/rag_eval_miss.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result["miss_cases"], f, ensure_ascii=False, indent=2)
    print(f"Miss cases saved to: {out_path}")


if __name__ == "__main__":
    main()
