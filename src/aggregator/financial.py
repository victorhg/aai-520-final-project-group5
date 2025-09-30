from ..worker.base_worker import BaseWorker

from datetime import datetime, timedelta
import yfinance as yf

class FinancialAggregator(BaseWorker):
    def execute(self, *inputs) -> str:
        """
            Fetch stock data using yfinance.
            
            Args:
                symbol: Stock ticker symbol
                period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            Returns:
                A dictionary with stock metrics and historical data.
        """
        symbol = inputs[0]
        period = inputs[1] if len(inputs) > 1 else "1mo"

        stock = yf.Ticker(symbol)

        # Get historical data and info
        hist = stock.history(period=period)
        info = stock.info
        
        # Calculate basic metrics
        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', hist['Close'].iloc[-2])
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        # Calculate volatility (30-day)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.tail(30).std() * (252 ** 0.5)  # Annualized
        
        # Get volume metrics
        avg_volume = hist['Volume'].tail(30).mean()
        current_volume = hist['Volume'].iloc[-1]
        
        result = {
            "symbol": symbol,
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
            "historical_data": hist.tail(30).to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }    
        return result