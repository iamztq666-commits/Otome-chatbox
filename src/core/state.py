import json, os
from dataclasses import dataclass, asdict

STATE_DIR = "./state"
os.makedirs(STATE_DIR, exist_ok=True)

@dataclass
class OtomeState:
    # 为了兼容你现有 prompt（还读取 state["trust"]），这里字段名先保留 trust
    # 语义：好感度/亲密度 affection
    trust: int = 20

    # 0初见 1熟悉 2心动 3默契恋人
    stage: int = 0

    # 0~2：回忆/剧情解锁等级（替代原 spoiler_level）
    memory_unlock: int = 0

    # 已解锁剧情/回忆节点
    unlocked_plots: list[str] = None

    # 可选：语气（gentle/playful/serious），不用也不影响
    tone: str = "gentle"

    def __post_init__(self):
        if self.unlocked_plots is None:
            self.unlocked_plots = []

        # 容错：老存档里可能还叫 spoiler_level
        # 如果从 json load 进来时没提供 memory_unlock，dataclass会用默认值
        # 这里主要是保证范围合法
        self.trust = int(self.trust)
        self.stage = int(self.stage)
        self.memory_unlock = int(self.memory_unlock)

        self.trust = max(0, min(100, self.trust))
        self.stage = max(0, min(3, self.stage))
        self.memory_unlock = max(0, min(2, self.memory_unlock))

def load_state(user_id: str) -> OtomeState:
    fp = os.path.join(STATE_DIR, f"{user_id}.json")
    if not os.path.exists(fp):
        return OtomeState()

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容旧字段：spoiler_level -> memory_unlock
    if "memory_unlock" not in data and "spoiler_level" in data:
        data["memory_unlock"] = data.get("spoiler_level", 0)

    # 兼容旧类名存档里可能包含多余字段，尽量忽略
    allowed = {"trust", "stage", "memory_unlock", "unlocked_plots", "tone"}
    clean = {k: v for k, v in data.items() if k in allowed}

    return OtomeState(**clean)

def save_state(user_id: str, state: OtomeState):
    fp = os.path.join(STATE_DIR, f"{user_id}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)

def update_trust_stage(state: OtomeState, user_msg: str) -> OtomeState:
    """
    乙游规则：
    - 用户表达喜欢/依赖/感谢/分享生活/脆弱情绪：加好感
    - 用户辱骂、强烈拒绝、操控：扣好感
    - 用户提出露骨性要求/越界：扣好感并降温
    - 阶段随好感推进：0-3
    """
    msg = (user_msg or "").strip()

    # 正向：陪伴、感谢、喜欢、分享生活与情绪
    pos = [
        "谢谢", "喜欢", "想你", "在意", "信你", "依靠", "需要你", "陪我",
        "今天", "刚刚", "我去", "我做了", "我遇到", "分享", "告诉你",
        "难受", "委屈", "焦虑", "压力", "失眠", "想哭", "崩溃", "不安",
        "抱抱", "牵手", "晚安", "早安"
    ]

    # 负向：辱骂、驱赶、攻击
    neg = [
        "滚", "烦", "闭嘴", "恶心", "讨厌", "垃圾", "废物", "去死", "拉黑", "别烦我"
    ]

    # 越界/露骨：扣分（乙游也要边界）
    boundary = [
        "做爱", "上床", "裸", "脱", "胸", "下面", "细节", "视频", "照片", "开房", "约炮"
    ]

    delta = 0

    if any(w in msg for w in pos):
        delta += 4
    if any(w in msg for w in neg):
        delta -= 12
    if any(w in msg for w in boundary):
        delta -= 10

    # 额外：如果用户明确表示拒绝/不舒服，应该降温而不是强行推进
    if any(w in msg for w in ["别这样", "不舒服", "不要", "停", "别提了"]):
        delta -= 6

    state.trust = max(0, min(100, int(state.trust) + delta))

    # 阶段阈值（你可自行调整）
    t = state.trust
    if t < 30:
        state.stage = 0
        state.memory_unlock = 0
    elif t < 55:
        state.stage = 1
        state.memory_unlock = 1
    elif t < 80:
        state.stage = 2
        state.memory_unlock = 2
    else:
        state.stage = 3
        state.memory_unlock = 2

    return state
