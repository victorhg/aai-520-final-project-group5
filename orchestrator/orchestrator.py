from datetime import datetime, timezone
import json
from aggregator.aggregator import Aggregator
from langchain_core.tools import tool
from langchain_core.runnables import RunnableParallel, RunnableLambda

# --- Ingestion Module ---
## I defined these functions here for now. They belong to ingestion module
def fetch_stock_data(symbol: str, interval: str = "1d") -> str:
    return "Apple stocks are going up!"

def fetch_news_data(query: str) -> str:
    return "Apple stocks are going up because they released a new product!"

# --- Orchestrator tools ---
ingestion_worker = RunnableParallel({
    "stock_data": RunnableLambda(lambda inputs: fetch_stock_data(inputs["symbol"], inputs.get("interval", "1d"))),
    "news_data": RunnableLambda(lambda inputs: fetch_news_data(inputs["symbol"])),
    # ...etc
})

# Remove @tool if you are not going to use an Agent to orchestrate You can call the functions directly in Orchestrator class methods.
@tool
def ingestion_worker_tool(symbol: str, interval: str = "1d") -> str:
    """Fetch all required data sources in parallel (stock, news, etc.)"""
    result = ingestion_worker.invoke({"symbol": symbol, "interval": interval})
    return json.dumps(result)

# You won't need this if you don't use an Agent to orchestrate
@tool
def aggregator_worker_tool(ingestion_output: str) -> str:
    """Aggregate and normalize collected financial data obtained from ingestion tools."""
    aggregator = Aggregator()
    return aggregator.execute() # pass data from ingestion_worker_tool() output


# --- Orchestrator Module ---
class Orchestrator:
    def __init__(self):
        self.tools = [
            ingestion_worker_tool,  # Parallel ingestion of all data sources
            aggregator_worker_tool,  # Data aggregation and normalization
        ]

    def execute(self, symbol: str, instructions: str):
        # Placeholder for orchestrator execution logic
        # here add the logic to use the tools in sequence or as needed.
        # You could use an agent to orchestrate the calls based on instructions, or
        # just call the tools directly in sequence. If so, remove @tool decorators above.
        return None

def run_investment_analysis(symbol: str, instructions: str):
    """Entrypoint to run the orchestrator with given stock symbol and instructions."""
    orchestrator = Orchestrator()
    orchestrator.execute(symbol, instructions)

if __name__ == "__main__":
    # run with python3 -m orchestrator.orchestrator
    orchestrator = Orchestrator()
    orchestrator.run_investment_analysis("AAPL", "Analyze the potential for AAPL stock for the past 10 days and recommend buy/sell/hold.")
