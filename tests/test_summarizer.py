from datetime import datetime, timedelta
import pandas as pd
from src.summarizer.summarizer import SummarizerWorker

def _toy_news(rows=8):
    now = datetime.now()
    return pd.DataFrame({
        "title": [f"T{i}" for i in range(rows)],
        "source": ["demo"] * rows,
        "published": [now - timedelta(days=i // 2) for i in range(rows)]
    })

def test_execute_returns_expected_keys_without_confidence():
    w = SummarizerWorker()
    out = w.execute({"symbol": "TEST", "raw_news": _toy_news(8), "window": 7})
    # Required keys for the new contract
    assert set(["symbol", "summary", "routed_notes", "artifacts", "memory_writes"]).issubset(out.keys())
    # Explicitly ensure confidence is gone
    assert "confidence" not in out

def test_routing_has_known_buckets():
    w = SummarizerWorker()
    out = w.execute({"symbol": "TEST", "raw_news": _toy_news(8)})
    buckets = out["routed_notes"]
    # New expected routing buckets
    assert set(buckets.keys()) == {"earnings", "macro", "company"}
    assert all(isinstance(v, list) for v in buckets.values())
