import re
from dataclasses import dataclass
from typing import List

@dataclass
class PlotDoc:
    plot_id: str
    title: str
    stage_min: int
    trust_min: int
    spoiler_level: int
    text: str
    keywords: list[str]

def _grab_int(block: str, key: str, default: int = 0) -> int:
    m = re.search(rf"{key}\s*:\s*(\d+)", block)
    return int(m.group(1)) if m else default

def _grab_title(block: str, default: str) -> str:
    m = re.search(r"###\s*标题\s*\n(.+)", block)
    return m.group(1).strip() if m else default

def _grab_keywords(block: str) -> list[str]:
    # 允许你写“用户提到：xxx/yyy/zzz”或“用户提到‘xxx’”
    m = re.search(r"用户提到[^\n]*[:：]\s*(.+)", block)
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r"[\/、，,\s]+", raw)
    return [p.strip("“”\"'.,。!！?？:：") for p in parts if p.strip()]

def load_plot_docs(plot_md_path: str) -> List[PlotDoc]:
    text = open(plot_md_path, "r", encoding="utf-8").read()

    # 按 PLOT 节点切块：以 "## [PLOT_ID: xxx]" 开始
    blocks = re.split(r"\n(?=##\s*\[PLOT_ID:)", text)
    docs: List[PlotDoc] = []

    for b in blocks:
        m = re.search(r"\[PLOT_ID:\s*([^\]]+)\]", b)
        if not m:
            continue
        plot_id = m.group(1).strip()
        stage_min = _grab_int(b, "stage_min", 0)
        trust_min = _grab_int(b, "trust_min", 0)
        spoiler_level = _grab_int(b, "spoiler_level", 0)
        title = _grab_title(b, plot_id)
        keywords = _grab_keywords(b)

        # 入库文本：把这个 plot 的全部内容作为可检索正文
        body = b.strip()
        docs.append(PlotDoc(
            plot_id=plot_id,
            title=title,
            stage_min=stage_min,
            trust_min=trust_min,
            spoiler_level=spoiler_level,
            text=body,
            keywords=keywords
        ))
    return docs
