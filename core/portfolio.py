"""长期配置：核心 + 卫星。"""
from core.logging import get_logger

logger = get_logger("core.portfolio")

ETF_MAP = {
    "半导体": "半导体ETF(512480)",
    "白酒": "白酒ETF(512690)",
    "医药": "医药ETF(512010)",
    "新能源": "新能源ETF(516160)",
    "证券": "证券ETF(512880)",
    "银行": "银行ETF(512800)",
    "光伏": "光伏ETF(515790)",
    "军工": "军工ETF(512660)",
    "消费": "消费ETF(159928)",
    "科技": "科技ETF(515000)",
}


def build_portfolio(sector_scores: list[dict], core_ratio: float = 0.7,
                    top_n: int = 4) -> dict:
    top = sorted(sector_scores, key=lambda s: s["score"], reverse=True)[:top_n]
    scores = [s["score"] for s in top]
    total = sum(scores) or 1.0
    satellite = []
    for s in top:
        weight = round((s["score"] / total) * (1.0 - core_ratio), 4)
        etf = ETF_MAP.get(s["name"], f"{s['name']}ETF")
        # name 保留板块名；etf 为映射到的具体 ETF（含代码）
        satellite.append({"name": s["name"], "etf": etf, "weight": weight})
    return {
        "core": {"name": "沪深300ETF(510300)", "weight": core_ratio,
                 "note": "宽基核心"},
        "satellite": satellite,
        "rebalance_rule": "权重偏离目标 5% 时再平衡",
        "summary": f"核心{int(core_ratio * 100)}% + 卫星{int((1 - core_ratio) * 100)}%",
    }
