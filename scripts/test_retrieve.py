from src.rag.retrieve import retrieve_for_otome

# 低亲密度：不该拿到 plot
pairs = retrieve_for_otome(
    query="我今天有点累，想有人陪我说说话",
    stage=0,
    affection=10,
    memory_unlock=0,
    k=6
)

print("=== low stage/affection ===")
for doc, meta in pairs:
    print(meta.get("type"), meta.get("title") or meta.get("plot_id"), meta.get("source"))

# 高亲密度：应该能拿到 plot
pairs2 = retrieve_for_otome(
    query="下雨了，我有点想你",
    stage=2,
    affection=60,
    memory_unlock=2,
    k=6
)

print("\n=== high stage/affection ===")
for doc, meta in pairs2:
    print(meta.get("type"), meta.get("title") or meta.get("plot_id"), meta.get("source"))
