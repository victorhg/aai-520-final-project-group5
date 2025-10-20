# src/summarizer/summarizer.py

from __future__ import annotations

from typing import Any, Dict, List

# --- Try to import a base worker; fall back gracefully if unavailable ---
try:
    from src.worker import BaseWorker  # type: ignore
except Exception:  # pragma: no cover
    class BaseWorker:  # minimal shim so this file works standalone
        def __init__(self, *args, **kwargs) -> None:
            pass

__all__ = ["SummarizerWorker"]

# --- Simple keyword routing for headlines ---
ROUTES: Dict[str, List[str]] = {
    "earnings": ["eps", "guidance", "revenue", "call", "forecast", "beat", "miss", "margin"],
    "macro":    ["fed", "rate", "cpi", "inflation", "jobs", "gdp", "unemployment", "yields", "oil"],
    # Anything else (including legacy buckets) will map to "company"
    "company":  ["product", "launch", "recall", "supply", "lawsuit", "merger", "partnership", "contract"],
}

# Legacy bucket names we collapse to "company"
LEGACY_TO_COMPANY = {"product", "m&a", "legal"}

def _route(text: str) -> str:
    t = (text or "").lower()
    for route, keys in ROUTES.items():
        if any(k in t for k in keys):
            return route
    return "company"


PROMPT_TEMPLATE = """You are a pragmatic equity analyst.
Goal: {goal}
Symbol: {symbol}

Context (recent daily stats + sampled headlines):
{context}

Write 5–8 concise bullets on likely near-term price *drivers* and 2 bullets on *key risks*.
Avoid hype; be specific. Include dates or sources inline when present.
"""


class SummarizerWorker(BaseWorker):
    """
    Summarizer worker that produces a compact analysis and headline routing.

    Contract:
      Input:
        - symbol: str
        - news_daily: optional pandas DataFrame with daily aggregates
        - raw_news: optional list[dict] or pandas DataFrame with 'title', 'source', 'date/published'
        - window: int days to show from news_daily (default 7)
        - analysis_goal: optional custom goal string

      Output (NO confidence field):
        - {
            "symbol": str,
            "summary": str,
            "routed_notes": {"earnings": list[str], "macro": list[str], "company": list[str]},
            "artifacts": {"prompt": str},
            "memory_writes": list[str]
          }
    """

    def __init__(self, name: str = "summarizer", role: str = "news_summary", model: str | None = None):
        # Defensive init in case BaseWorker signature differs
        try:
            super().__init__(name=name, role=role, model=model)  # type: ignore[misc]
        except TypeError:
            try:
                super().__init__()  # type: ignore[misc]
            except Exception:
                pass
            setattr(self, "name", name)
            setattr(self, "role", role)
            setattr(self, "model", model)

    # ----------------- Public API -----------------
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for the summarizer agent.

        NOTE: Confidence metric is fully removed (no 'confidence' in output).
        """
        symbol: str = inputs["symbol"]
        news_daily = inputs.get("news_daily")
        raw_news   = inputs.get("raw_news")
        window     = int(inputs.get("window", 7))
        goal       = inputs.get("analysis_goal", f"Next-week price drivers for {symbol}")

        context = self._format_context(news_daily, raw_news, window)
        routed  = self._route_headlines(raw_news)
        prompt  = PROMPT_TEMPLATE.format(goal=goal, symbol=symbol, context=context)

        # ---- Stubbed model call (replace with your LLM invocation) ----
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

    # ----------------- Helpers -----------------
    def _format_context(
        self,
        news_daily: Any,
        raw_news: Any,  # Union[List[dict], "pd.DataFrame", None] at runtime
        window: int,
    ) -> str:
        parts: List[str] = []

        # Daily aggregates (expects pandas-like DataFrame)
        if news_daily is not None and hasattr(news_daily, "tail") and len(news_daily) > 0:
            try:
                tail = news_daily.tail(window)
                parts.append("Daily sentiment (most recent first):")
                # reverse chronological for readability
                for idx, row in tail.iloc[::-1].iterrows():
                    parts.append(
                        f"- {idx}: count={int(row.get('news_count', 0))}, "
                        f"sent_mean={row.get('sent_mean', 0):+.3f}, "
                        f"decay={row.get('sent_decay', 0):+.3f}"
                    )
            except Exception:
                # stay robust if columns/format differ
                pass

        # Recent headlines
        if raw_news is not None:
            try:
                import pandas as pd  # local import to avoid hard dep at import time
                df = raw_news if isinstance(raw_news, pd.DataFrame) else pd.DataFrame(raw_news)
                # accept either 'date' or 'published'
                date_col = "date" if "date" in df.columns else ("published" if "published" in df.columns else None)
                cols = [c for c in [date_col, "source", "title"] if c]
                if cols:
                    parts.append("\nRecent headlines:")
                    for _, r in df.tail(min(12, len(df))).iloc[::-1].iterrows():
                        parts.append("- " + " | ".join(str(r.get(c, "")) for c in cols))
            except Exception:
                pass

        return "\n".join(parts) if parts else "(No news context available)"

    def _route_headlines(self, raw_news: Any) -> Dict[str, List[str]]:
        """
        Returns exactly three buckets: earnings, macro, company.
        Legacy buckets like 'product', 'm&a', 'legal' are normalized into 'company'.
        """
        routed: Dict[str, List[str]] = {"earnings": [], "macro": [], "company": []}

        if raw_news is None:
            return routed

        try:
            import pandas as pd  # local import to avoid hard dep at import time
            df = raw_news if isinstance(raw_news, pd.DataFrame) else pd.DataFrame(raw_news)
            # Collect to a temporary dict that may include legacy buckets
            temp: Dict[str, List[str]] = {"earnings": [], "macro": [], "company": [], "product": [], "m&a": [], "legal": []}
            for _, r in df.tail(50).iterrows():
                ttl = str(r.get("title", "")) or ""
                temp[_route(ttl)].append(ttl)

            # ---- Normalize to new buckets ----
            norm: Dict[str, List[str]] = {"earnings": [], "macro": [], "company": []}
            for k, v in temp.items():
                target = "company" if (k in LEGACY_TO_COMPANY or k not in norm) else k
                norm[target].extend(v)

            # Trim to a few examples per bucket
            return {k: vals[:5] for k, vals in norm.items()}
        except Exception:
            # On any parsing error, just return empty buckets (stable contract)
            return routed
