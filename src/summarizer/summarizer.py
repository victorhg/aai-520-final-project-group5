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

# --- Simple keyword routing for headlines ---
ROUTES = {
    "earnings": ["eps", "guidance", "revenue", "call", "forecast", "beat", "miss", "margin"],
    "macro":    ["fed", "rate", "cpi", "inflation", "jobs", "gdp", "unemployment", "yields", "oil"],
    "company":  ["product", "launch", "recall", "supply", "lawsuit", "merger", "partnership", "contract"],
}

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
    def __init__(self, name: str = "summarizer", role: str = "news_summary", model: str | None = None):
        """
        Defensive init:
        - Tries super().__init__(name=..., role=..., model=...).
        - If parent __init__ takes no args or is missing, call it without args (if present)
          and set attributes locally as a fallback.
        """
        # Try the most specific signature first
        try:
            super().__init__(name=name, role=role, model=model)  # type: ignore[misc]
        except TypeError:
            # Parent __init__ takes no args (or doesn't define one)
            try:
                super().__init__()  # type: ignore[misc]
            except Exception:
                pass
            # Fallback: ensure attributes exist on self
            setattr(self, "name", name)
            setattr(self, "role", role)
            setattr(self, "model", model)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for the summarizer agent.

        Fix included:
        - If `news_daily` is missing but `raw_news` is provided, derive a daily
          aggregate so confidence doesn't default to 0.50.
        - Count rows per day with groupby().size() so missing titles don't drop counts.
        """
        import pandas as pd

        symbol: str = inputs["symbol"]
        news_daily = inputs.get("news_daily")
        raw_news   = inputs.get("raw_news")
        window     = int(inputs.get("window", 7))
        goal       = inputs.get("analysis_goal", f"Next-week price drivers for {symbol}")

        # ---- Build news_daily from raw_news if not provided ----
        if news_daily is None and raw_news is not None:
            try:
                df = raw_news if isinstance(raw_news, pd.DataFrame) else pd.DataFrame(raw_news)

                # Normalize/derive a date column
                if "published" in df.columns:
                    df["published"] = pd.to_datetime(df["published"], errors="coerce")
                    df["date"] = df["published"].dt.date
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                else:
                    df["date"] = pd.Timestamp.today().date()  # fallback to today if no timestamp present

                # Robust daily aggregation (counts all rows, even if title is NaN)
                news_daily = (
                    df.groupby("date")
                      .size()
                      .rename("news_count")
                      .to_frame()
                      .reset_index()
                      .assign(sent_mean=0.0, sent_decay=0.0)
                      .set_index("date")
                )
            except Exception as e:
                print(f"[WARN] Could not derive news_daily from raw_news: {e}")
                news_daily = None
        # --------------------------------------------------------

        context = self._format_context(news_daily, raw_news, window)
        routed  = self._route_headlines(raw_news)
        prompt  = PROMPT_TEMPLATE.format(goal=goal, symbol=symbol, context=context)

        # NOTE: Replace this stub with your actual LLM call when ready.
        summary_text = (
            "(Stubbed summary — replace with your LLM call)\n"
            + prompt
            + "\n- Headlines cluster around a few catalysts; monitor official updates.\n"
              "- Tone is slightly positive; momentum sensitive to macro prints.\n"
              "- Risks: guidance/margin pressure; policy surprises."
        )

        confidence = self._confidence_from_news(news_daily, window)
        memory_writes = [
            f"[{symbol}] {window}d summary (conf={confidence:.2f})",
            f"[{symbol}] Routes: " + ", ".join([k for k, v in routed.items() if v])
        ]

        return {
            "symbol": symbol,
            "summary": summary_text,
            "routed_notes": routed,
            "confidence": confidence,
            "artifacts": {"prompt": prompt},
            "memory_writes": memory_writes,
        }

    # ----------------- Helpers -----------------
    def _format_context(
        self,
        news_daily,
        raw_news: Union[List[dict], "pd.DataFrame", None],
        window: int
    ) -> str:
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
                df = raw_news if isinstance(raw_news, pd.DataFrame) else pd.DataFrame(raw_news)
                ts = "published" if "published" in df.columns else ("date" if "date" in df.columns else None)
                if ts:
                    df = df.sort_values(by=ts).tail(12)
                parts.append("Recent headlines:")
                for _, r in df.iterrows():
                    ttl = str(r.get("title", ""))[:160]
                    src = r.get("source", "") or "news"
                    dt  = r.get("published", r.get("date", ""))
                    parts.append(f"- [{dt}] ({src}) {ttl}")
            except Exception:
                pass

        return "\n".join(parts) if parts else "No recent news."

    def _route_headlines(self, raw_news) -> dict:
        routed = {"earnings": [], "macro": [], "company": []}
        if raw_news is None:
            return routed
        try:
            import pandas as pd
            df = raw_news if isinstance(raw_news, pd.DataFrame) else pd.DataFrame(raw_news)
            for _, r in df.tail(50).iterrows():
                ttl = str(r.get("title", "")) or ""
                routed[_route(ttl)].append(ttl)
        except Exception:
            pass
        # Trim to a few examples per bucket
        return {k: v[:5] for k, v in routed.items()}

    def _confidence_from_news(self, news_daily, window: int) -> float:
        """
        Confidence rises modestly with average daily news volume.
        If news_daily is missing or malformed, fall back to 0.50.
        """
        if news_daily is None or not hasattr(news_daily, "tail") or len(news_daily) == 0:
            return 0.50
        try:
            avg_cnt = float(news_daily.tail(window)["news_count"].mean())
            return round(min(1.0, 0.5 + 0.05 * avg_cnt), 2)
        except Exception:
            return 0.50
