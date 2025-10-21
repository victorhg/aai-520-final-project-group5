# --- Standard Library ---
import os
from datetime import datetime
from typing import Dict, Any, List

# --- Workers ---
from src.memory_analyzer.memory_analyzer import MemoryAnalyzer
from src.ingestion.ingestion import Ingestion
from src.summarizer.summarizer import SummarizerWorker
from src.memory.memory import MemoryWorker
from src.evaluator_optimizer.evaluator_optimizer import EvaluatorOptimize

# --- Orchestrator Agent ---
class Orchestrator:
    """
    Coordinates the workflow of all workers:
    1. Retrieve memory for symbol
    2. Analyze if memory is sufficient using MemoryAnalyzer
    3. If sufficient: use cached optimized summary, else: fetch fresh data via Ingestion
    4. Summarizer: generate summary from data
    5. Memory: store summary and metadata
    6. EvaluatorOptimizer: refine the summary iteratively (skipped if already optimized)
    """

    # -----------------------------
    # Worker Initialization
    # -----------------------------
    def __init__(self):
        self.workers = {
            "analyzer": MemoryAnalyzer(),
            "ingestion": Ingestion(),
            "summarizer": SummarizerWorker(),
            "memory": MemoryWorker(),
            "evaluator_optimizer": EvaluatorOptimizer()
        }

    # -----------------------------
    # Planning and Routing Execution
    # -----------------------------
    def execute(self, symbol: str, instructions: str) -> str:
        """Runs the full workflow and returns a formatted Markdown summary"""
        
        print(f"\n{'='*60}")
        print(f" ORCHESTRATOR STARTING")
        print(f"Symbol: {symbol}")
        print(f"Query: {instructions}")
        print(f"{'='*60}\n")

        # --- Step 1: Retrieve previous memories for this symbol ---
        previous_memories = self.workers["memory"].execute("retrieve_by_symbol", symbol)
        print(f" Retrieved {len(previous_memories)} previous memory entries for {symbol}")
        
        # --- Step 2: Format memories into a snapshot for analysis ---
        memory_snapshot = self._format_memory_snapshot(previous_memories, symbol)
        
        # --- Step 3: Analyze if existing memory is sufficient ---
        print(f"\n Analyzing memory sufficiency...")
        analysis_result = self.workers["analyzer"].execute(memory_snapshot, instructions)
        
        print(f"Analysis Result:")
        print(f"  - Data Sufficient: {analysis_result['is_sufficient']}")
        print(f"  - Requires Fresh Data: {analysis_result['requires_fresh_data']}")
        if analysis_result.get('missing_information'):
            print(f"  - Missing Info: {', '.join(analysis_result['missing_information'][:3])}")

        # --- Step 4: Decide whether to use memory or fetch fresh data ---
        optimized_memories = [
            mem for mem in previous_memories
            # Check summary with optimized tag
            if "summary" in mem.get("tags", []) and "optimized" in mem.get("tags", [])
        ]
        # Verify optimized memories are sufficient
        if optimized_memories and analysis_result["is_sufficient"] and not analysis_result["requires_fresh_data"]:
            # Use cached summary only if memory is sufficient and fresh
            print(f"\n Using previously optimized memory summary")
            most_recent_optimized = optimized_memories[-1]
    
            summary_result = {
                "symbol": symbol,
                "summary": most_recent_optimized.get("text", ""),
                "routed_notes": {},
                "artifacts": {"source": "memory"},
                "memory_writes": []
            }
            # Set to skip EvaluatorOptimizer
            skip_optimizer = True
            ingestion_result = {}  # no new ingestion needed
        else:
            # Memory insufficient, stale, or no optimized summary exists
            print(f"\n Fetching fresh data (memory insufficient, stale, or not optimized)")
            
            ingestion_result = self.workers["ingestion"].execute(symbol)
            
            news_articles = (ingestion_result.get("news_data") or {}).get("articles", [])
            
            summary_result = self.workers["summarizer"].execute({
                "symbol": symbol,
                "raw_news": news_articles,
                "window": 7,
                "analysis_goal": instructions
            })
            # Store new intitial summary in memory with summary tag
            for note in summary_result.get("memory_writes", []):
                self.workers["memory"].execute("add", note, [symbol, "summary"])
            # Set to false to run EvaluatorOptimizer
            skip_optimizer = False

        # --- Step 5: Evaluator-Optimizer Workflow ---
        if not skip_optimizer:
            evaluator = self.workers["evaluator_optimizer"]
            initial_state = {
                "symbol": symbol,
                "instructions": instructions,
                "context": ingestion_result,
                "summary": summary_result["summary"],
                "feedback": "",
                "grade": "",
                "quality_score": 0.0,
                "issues": [],
                "iteration": 0,
                "max_iterations": evaluator.max_iterations,
                "history": []
            }
            final_result = evaluator.workflow.invoke(initial_state)
            final_summary_text = final_result["summary"]
            
            # Store optimized summary in memory
            memory_note = f"[{symbol}] Final Optimized Summary (Query: {instructions[:50]}...)"
            timestamp = datetime.now().isoformat()
            full_memory_entry = f"{memory_note}\nTimestamp: {timestamp}\n\n{final_summary_text}"
            # Store final optimized summary in memory with symbol, "summary" and "optimized" tags
            self.workers["memory"].execute("add", full_memory_entry, [symbol, "summary", "optimized"])
        else:
            # Already optimized, skip optimizer
            final_summary_text = summary_result["summary"]

        print(f"\n{'='*60}")
        print(f" ORCHESTRATOR COMPLETE")
        print(f"{'='*60}\n")

        return final_summary_text  # returns plain Markdown string

    # -----------------------------
    # Helper Methods - formatting memory snapshot
    # -----------------------------
    def _format_memory_snapshot(self, memories: List[Dict[str, Any]], symbol: str) -> str:
        """
        Format an array of memory records into a single string snapshot for analysis.
        Only include optimized summaries for memory-sufficient evaluation.
        """
        if not memories:
            return ""
        
        snapshot_parts = [f"# Memory Data for {symbol}\n"]
        
        # Group memories by type (summary, routes, other)
        summaries = [
            f"## Entry from {mem.get('timestamp','N/A')}\n{mem.get('text','')}\n"
            for mem in memories
            if 'summary' in mem.get('tags', []) and 'optimized' in mem.get('tags', [])
        ]
        routes = []
        other = []
        
        # Iteratre through each memory to categorize
        for mem in memories:
            text = mem.get('text', '')
            timestamp = mem.get('timestamp', 'N/A')
            
            if 'Routes:' in text:
                routes.append(f"- {text} (at {timestamp})")
            elif 'summary' not in mem.get('tags', []):
                other.append(f"- {text} (at {timestamp})")
        
        # If previous summaries exits, add header to include last 2
        if summaries:
            snapshot_parts.append("\n## Previous Summaries\n")
            snapshot_parts.extend(summaries[-2:])
        # If previous routing notes exist, add header to include last 3
        if routes:
            snapshot_parts.append("\n## Previous Routing Information\n")
            snapshot_parts.extend(routes[-3:])
        # If previous other notes exist, add section header to include last 5
        if other:
            snapshot_parts.append("\n## Other Notes\n")
            snapshot_parts.extend(other[-5:])
        
        return "\n".join(snapshot_parts)
        
    # Retrieve most recent optimized summary tags
    def _extract_from_memory(self, memories: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Extract ingestion-like data structure from memory records.
        Only retrieves fully optimized summaries.
        """
        most_recent = None
        for mem in reversed(memories):
            # Check memory entry tagged for summary and optimized
            if 'summary' in mem.get('tags', []) and 'optimized' in mem.get('tags', []):
                most_recent = mem
                break
        
        # Return placeholder if no optimized summary found
        if not most_recent:
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "financial_data": None,
                "news_data": None,
                "errors": [{"source": "memory", "note": "Using cached data"}],
                "status": "success"
            }
        # Return memory structure for most recent optimized summary
        return {
            "symbol": symbol,
            "timestamp": most_recent.get('timestamp', datetime.now().isoformat()),
            "financial_data": {
                "source": "memory",
                "summary": most_recent.get('text', '')
            },
            "news_data": {
                "source": "memory",
                "articles": []
            },
            "errors": [],
            "status": "success"
        }
# -----------------------------
# Helper Function to Run Analysis
# -----------------------------
def run_investment_analysis(symbol: str, instructions: str) -> str:
    """
    Convenience function to run a complete investment analysis.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA", "NVDA")
        instructions: User's query or analysis request
        
    Returns:
        Final optimized summary as Markdown string
        
    Example:
        result = run_investment_analysis("AAPL", "Should I buy Apple stock now?")
        display(Markdown(result))
    """
    orchestrator = Orchestrator()
    
    return orchestrator.execute(symbol, instructions)
