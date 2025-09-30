from datetime import datetime, timezone
import json
from src.aggregator.aggregator import Aggregator
from src.ingestion.ingestion import Ingestion
from langchain_core.tools import tool
from langchain_core.runnables import RunnableParallel, RunnableLambda

# --- Orchestrator Module ---
class Orchestrator:
    def __init__(self):
        self.workers = {
            "ingestion": Ingestion(),  # Parallel ingestion of all data sources
            "aggregation": Aggregator(),  # Data aggregation and normalization
        }

    def execute(self, symbol: str, instructions: str):
        # here add the logic to use the tools in sequence or as needed.
        # TODO: We need to know from the instructors in this should be agentic, or if it can be programmatic
        # I imagine we need an OutputAgent here as well to polish the final output
        # If agentic, these would have to be @tools

        # example
        print(self.workers["ingestion"].execute())
        print(self.workers["aggregation"].execute())
        return None

def run_investment_analysis(symbol: str, instructions: str):
    """Entrypoint to run the orchestrator with given stock symbol and instructions."""
    orchestrator = Orchestrator()
    orchestrator.execute(symbol, instructions)

if __name__ == "__main__":
    # run with python3 -m orchestrator.orchestrator
    run_investment_analysis("AAPL", "Analyze the potential for AAPL stock for the past 10 days and recommend buy/sell/hold.")
