# tests/test_summarizer.py
# Verifies confidence > 0.50 when raw_news is present (i.e., news_daily is derived),
# and == 0.50 when there's no news.

from datetime import datetime, timedelta
import pandas as pd

from src.summarizer.summarizer import SummarizerWorker

def _toy_news(rows=8):
    now = datetime.now()
    return pd.DataFrame({
        "title": [f"T{i}" for i in range(rows-1)] + [None],  # include a None to test robust counting
        "source": ["demo"] * rows,
        "published": [now - timedelta(days=i // 2) for i in range(rows)]
    })

def test_confidence_builds_from_raw_news():
    w = SummarizerWorker()
    out = w.execute({"symbol": "TEST", "raw_news": _toy_news(8), "window": 7})
    assert out["confidence"] > 0.50

def test_confidence_falls_back_when_no_news():
    w = SummarizerWorker()
    out = w.execute({"symbol": "TEST", "raw_news": None, "window": 7})
    assert out["confidence"] == 0.50
