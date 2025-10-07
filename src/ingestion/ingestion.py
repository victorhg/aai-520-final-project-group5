"""
Main Ingestion Coordinator
Orchestrates parallel data fetching from financial and news sources (simplified)
"""

from src.worker.base_worker import BaseWorker
from src.ingestion.financial_data import FinancialDataIngestion
from src.ingestion.news_data import NewsDataIngestion
from typing import Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class Ingestion(BaseWorker):
    """
    Main Ingestion Coordinator that fetches data from financial and news sources in parallel.
    
    Responsibilities:
    - Coordinate parallel data fetching from financial and news sources
    - Combine results into single bundle
    - Handle partial failures gracefully
    - Track errors from each source
    - Return complete data bundle
    """
    
    def __init__(self):
        """Initialize data ingestors."""
        self.financial_ingestor = FinancialDataIngestion()
        self.news_ingestor = NewsDataIngestion()
    
    def execute(self, *inputs) -> Dict[str, Any]:
        """
        Execute parallel data ingestion from all sources.
        
        Args:
            inputs[0] (str): Stock ticker symbol (e.g., "AAPL")
            inputs[1] (str, optional): Time period for historical data (default: "1mo")
            inputs[2] (int, optional): Number of news articles (default: 10)
        
        Returns:
            dict: Complete data bundle with financial and news data
        """
        try:
            # Extract parameters
            symbol = inputs[0] if len(inputs) > 0 else "AAPL"
            period = inputs[1] if len(inputs) > 1 else "1mo"
            news_limit = inputs[2] if len(inputs) > 2 else 10
            
            # Execute parallel fetching
            results = self._execute_parallel(symbol, period, news_limit)
            
            # Combine results
            bundle = self._combine_results(symbol, results)
            
            return bundle
            
        except Exception as e:
            return {
                "symbol": inputs[0] if len(inputs) > 0 else "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "financial_data": None,
                "news_data": None,
                "errors": [{"source": "ingestion_coordinator", "error": str(e)}],
                "status": "error"
            }
    
    def _execute_parallel(self, symbol: str, period: str, news_limit: int) -> Dict[str, Any]:
        """
        Execute all ingestion tasks in parallel.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period for historical data
            news_limit: Number of news articles
            
        Returns:
            Dictionary with results from all sources
        """
        results = {
            "financial": None,
            "news": None
        }
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks
            future_to_source = {
                executor.submit(self.financial_ingestor.execute, symbol, period): "financial",
                executor.submit(self.news_ingestor.execute, symbol, news_limit): "news"
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    results[source] = result
                except Exception as e:
                    results[source] = {
                        "source": source,
                        "data": None,
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
        
        return results
    
    def _combine_results(self, symbol: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from all ingestors into single bundle.
        
        Args:
            symbol: Stock ticker symbol
            results: Dictionary with results from each ingestor
            
        Returns:
            Combined data bundle
        """
        errors = []
        
        # Extract financial data
        financial_result = results.get("financial", {})
        financial_data = financial_result.get("data") if financial_result.get("status") == "success" else None
        if financial_result.get("error"):
            errors.append({"source": "financial", "error": financial_result.get("error")})
        
        # Extract news data
        news_result = results.get("news", {})
        news_data = news_result.get("data") if news_result.get("status") in ["success", "partial_success"] else None
        if news_result.get("error"):
            errors.append({"source": "news", "error": news_result.get("error")})
        
        # Determine overall status
        status = self._determine_status(results)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "financial_data": financial_data,
            "news_data": news_data,
            "errors": errors,
            "status": status
        }
    
    def _determine_status(self, results: Dict[str, Any]) -> str:
        """
        Determine overall ingestion status based on results.
        
        Args:
            results: Dictionary with results from each ingestor
            
        Returns:
            Status string: "success", "partial_success", or "error"
        """
        success_count = 0
        total_count = len(results)
        
        for source, result in results.items():
            if result and result.get("status") in ["success", "partial_success"]:
                success_count += 1
        
        if success_count == total_count:
            return "success"
        elif success_count > 0:
            return "partial_success"
        else:
            return "error"
