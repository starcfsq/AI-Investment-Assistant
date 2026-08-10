TREND_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

SECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "logic": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

STOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "report": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

PORTFOLIO_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "rebalance": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "references": {"type": "array", "items": {"type": "object"}},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}
