# core/adapters.py
# 目的：把你仓库里各种版本的函数名统一成一个稳定接口

# --- RAG ---
try:
    from rag.retrieve import retrieve_for_otome as _retrieve
except Exception:
    from rag.retrieve import retrieve_canon as _retrieve

def retrieve_for_otome(query, stage, affection, memory_unlock, k=6):
    # 有些实现叫 trust/spoiler_level
    try:
        return _retrieve(query=query, stage=stage, affection=affection, memory_unlock=memory_unlock, k=k)
    except TypeError:
        return _retrieve(query=query, stage=stage, trust=affection, spoiler_level=memory_unlock, k=k)

# --- Prompt ---
try:
    from core.prompt import build_messages
except Exception:
    from core.prompt import build_prompt as build_messages  # 兜底：把 string 当 messages 用

# --- Unlock ---
try:
    from story.unlock import unlock_memories
except Exception:
    try:
        from story.unlock import unlock_new_plots as unlock_memories
    except Exception:
        unlock_memories = None