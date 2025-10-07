"""
NLP-5: Financial Data Ingestion
Fetch stock data from Yahoo Finance API (yfinance)
"""

from src.worker.base_worker import BaseWorker
from typing import Dict, Any
from datetime import datetime
import yfinance as yf


class FinancialDataIngestion(BaseWorker):
    """
    Fetches financial data from Yahoo Finance.
    
    Responsibilities:
    - Fetch historical OHLCV (Open, High, Low, Close, Volume) data
    - Fetch company information and fundamentals
    - Calculate basic metrics from raw data
    - Handle API errors gracefully
    - Return structured financial data
    """
    
    def execute(self, *inputs) -> Dict[str, Any]:
        """
        Fetch financial data for a given stock symbol.
        
        Args:
            inputs[0] (str): Stock ticker symbol (e.g., "AAPL")
            inputs[1] (str, optional): Time period (default: "1mo")
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        
        Returns:
            dict: Financial data bundle containing:
                - symbol: Stock ticker
                - price metrics: Current price, changes, highs/lows
                - volume metrics: Current and average volume
                - volatility: Calculated volatility
                - fundamentals: P/E ratio, market cap, beta, etc.
                - company info: Sector, industry, summary
                - historical_data: Recent OHLCV records
                - status: Success or error status
        """
        try:
            symbol = inputs[0]
            period = inputs[1] if len(inputs) > 1 else "1mo"

            stock = yf.Ticker(symbol)

            # Get historical data and info
            hist = stock.history(period=period)
            
            if hist.empty:
                return {
                    "source": "yahoo_finance",
                    "symbol": symbol,
                    "data": None,
                    "status": "error",
                    "error": f"No data found for symbol {symbol}",
                    "timestamp": datetime.now().isoformat()
                }
            
            info = stock.info
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
            
            # Calculate volatility (30-day annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.tail(30).std() * (252 ** 0.5) if len(returns) > 0 else 0
            
            # Get volume metrics
            avg_volume = hist['Volume'].tail(30).mean()
            current_volume = hist['Volume'].iloc[-1]
            
            result = {
                "source": "yahoo_finance",
                "symbol": symbol,
                "data": {
                    "current_price": float(current_price),
                    "price_change": float(price_change),
                    "price_change_pct": float(price_change_pct),
                    "volume": int(current_volume),
                    "avg_volume_30d": float(avg_volume),
                    "volatility_30d": float(volatility),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("forwardPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "beta": info.get("beta"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "company_summary": info.get("longBusinessSummary", "")[:500],
                    "historical_data": hist.tail(30).to_dict('records')
                },
                "status": "success",
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "source": "yahoo_finance",
                "symbol": inputs[0] if len(inputs) > 0 else "UNKNOWN",
                "data": None,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

