# src/summarizer/summarizer.py
from __future__ import annotations
from typing import Dict, Any, List, Union

# --- Resolve BaseWorker import across common repo layouts ---
try:
    # (A) src/worker/base_worker.py
    from ..worker.base_worker import BaseWorker
except Exception:
    try:
        # (B) src/base_worker.py
        from ..base_worker import BaseWorker
    except Exception:
        # (C) base_worker.py at repo root (no src/)
        from base_worker import BaseWorker

PROMPT_TEMPLATE = """You are a pragmatic equity analyst.
Goal: {goal}
Symbol: {symbol}

Context (recent daily stats + sampled headlines):
{context}

Write 5–8 concise bullets on likely near-term price *drivers* and 2 bullets on *key risks*.
Avoid hype; be specific. Include dates or sources inline when present.
"""

class SummarizerWorker(BaseWorker):
    def __init__(self, name: str = "summarizer", role: str = "news_summary", model: str | None = None):
        """
        Defensive init so this worker works whether BaseWorker stores attributes in __init__
        or we need to set them here to align with the shared interface.
        """
        try:
            super().__init__(name=name, role=role, model=model)
        except Exception:
            setattr(self, "name", name)
            setattr(self, "role", role)
            setattr(self, "model", model)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for the summarizer agent.

        CHANGE: removed confidence metric entirely.
        - No 'confidence' field in the return payload
        - No dependence on news_daily volume for any score
        """
        symbol: str = inputs["symbol"]
        news_daily = inputs.get("news_daily")
        raw_news   = inputs.get("raw_news")
        window     = int(inputs.get("window", 7))
        goal       = inputs.get("analysis_goal", f"Next-week price drivers for {symbol}")

        # Build the textual context block shown to the LLM
        context = self._format_context(news_daily, raw_news, window)
        routed  = self._route_headlines(raw_news)
        prompt  = PROMPT_TEMPLATE.format(goal=goal, symbol=symbol, context=context)

        # NOTE: Replace this stub with your actual LLM call when integrated.
        summary_text = (
            "(Stubbed summary — replace with your LLM call)\n"
            + prompt
            + "\n- Headlines cluster around a few catalysts; monitor official updates.\n"
              "- Tone is slightly positive; momentum sensitive to macro prints.\n"
              "- Risks: guidance/margin pressure; policy surprises."
        )

        memory_writes = [
            f"[{symbol}] {window}d summary",
            f"[{symbol}] Routes: " + ", ".join([k for k, v in routed.items() if v])
        ]

        return {
            "symbol": symbol,
            "summary": summary_text,
            "routed_notes": routed,
            "artifacts": {"prompt": prompt},
            "memory_writes": memory_writes,
        }

    # ------------------------- helpers -------------------------
    def _format_context(self, news_daily, raw_news, window: int) -> str:
        """
        Compose a compact, human-readable context out of daily aggregates
        plus a short sample of recent headlines.
        """
        parts: List[str] = []

        # Daily aggregates
        if news_daily is not None and hasattr(news_daily, "tail") and len(news_daily) > 0:
            tail = news_daily.tail(window)
            parts.append("Daily sentiment (most recent first):")
            for idx, row in tail.iloc[::-1].iterrows():
                parts.append(
                    f"- {idx}: count={int(row.get('news_count', 0))}, "
                    f"sent_mean={row.get('sent_mean', 0):+.3f}, decay={row.get('sent_decay', 0):+.3f}"
                )

        # Recent headlines
        if raw_news is not None:
            try:
                import pandas as pd
                if not isinstance(raw_news, pd.DataFrame):
                    # attempt to coerce list[dict] to DataFrame if needed
                    raw_news = pd.DataFrame(raw_news)
                # pick a few latest headlines
                cols = [c for c in ["date", "source", "title"] if c in raw_news.columns]
                if cols:
                    parts.append("\nRecent headlines:")
                    for _, r in raw_news.tail(min(12, len(raw_news))).iloc[::-1].iterrows():
                        parts.append("- " + " | ".join(str(r.get(c, "")) for c in cols))
            except Exception:
                pass

        return "\n".join(parts) if parts else "(No news context available)"

    def _route_headlines(self, raw_news) -> Dict[str, List[str]]:
        """
        Naive keyword routing so downstream agents can decide follow-ups.
        """
        buckets = {
            "earnings": ["earnings", "eps", "guidance"],
            "product":  ["launch", "recall", "feature", "chip", "software"],
            "legal":    ["lawsuit", "investigation", "settlement", "fine"],
            "macro":    ["rates", "inflation", "jobs", "cpi", "ppi", "fed"],
            "m&a":      ["acquire", "acquisition", "merger", "deal"],
        }
        routed: Dict[str, List[str]] = {k: [] for k in buckets}
        try:
            import pandas as pd
            df = raw_news
            if df is None:
                return routed
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            titles = df.get("title") if "title" in df.columns else df.get("headline")
            if titles is None:
                return routed
            for t in titles.tail(min(25, len(df))).fillna("").tolist():
                low = t.lower()
                for k, kws in buckets.items():
                    if any(kw in low for kw in kws):
                        routed[k].append(t)
        except Exception:
            pass
        # Trim to a few examples per bucket
        return {k: v[:5] for k, v in routed.items()}
