"""
Financial data collection tools for the Investment Research Agent.
"""

import os
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import feedparser
import json

from ..agents.base_agent import Tool


class StockDataTool(Tool):
    """Tool for collecting stock price data and basic financial metrics."""
    
    @property
    def name(self) -> str:
        return "stock_data"
    
    @property
    def description(self) -> str:
        return "Retrieves stock price data, financial metrics, and company information"
    
    def execute(self, symbol: str, period: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        Fetch stock data using yfinance.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        try:
            stock = yf.Ticker(symbol)
            
            # Get historical data
            hist = stock.history(period=period)
            
            # Get stock info
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
            
        except Exception as e:
            return {"error": f"Failed to fetch stock data for {symbol}: {str(e)}"}


class NewsDataTool(Tool):
    """Tool for collecting financial news and sentiment."""
    
    @property
    def name(self) -> str:
        return "news_data"
    
    @property
    def description(self) -> str:
        return "Retrieves financial news articles and RSS feeds related to stocks"
    
    def execute(self, symbol: str, query: str = "", limit: int = 20, **kwargs) -> Dict[str, Any]:
        """
        Fetch news data from multiple sources.
        
        Args:
            symbol: Stock ticker symbol
            query: Additional search query
            limit: Maximum number of articles
        """
        try:
            news_sources = []
            
            # Yahoo Finance RSS feed
            try:
                yahoo_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                yahoo_feed = feedparser.parse(yahoo_rss)
                
                for entry in yahoo_feed.entries[:limit//2]:
                    news_sources.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", ""),
                        "source": "Yahoo Finance"
                    })
            except Exception as e:
                print(f"Error fetching Yahoo Finance news: {e}")
            
            # Google Finance RSS (alternative approach)
            try:
                google_rss = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
                google_feed = feedparser.parse(google_rss)
                
                for entry in google_feed.entries[:limit//2]:
                    news_sources.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", ""),
                        "source": "Google News"
                    })
            except Exception as e:
                print(f"Error fetching Google News: {e}")
            
            # If we have an API key, use News API
            news_api_key = os.getenv("NEWS_API_KEY")
            if news_api_key:
                try:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": f"{symbol} stock OR {query}",
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": limit//2,
                        "apiKey": news_api_key
                    }
                    
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        for article in data.get("articles", []):
                            news_sources.append({
                                "title": article["title"],
                                "link": article["url"],
                                "published": article["publishedAt"],
                                "summary": article.get("description", ""),
                                "source": article["source"]["name"]
                            })
                except Exception as e:
                    print(f"Error fetching News API data: {e}")
            
            result = {
                "symbol": symbol,
                "articles": news_sources[:limit],
                "total_articles": len(news_sources),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to fetch news data for {symbol}: {str(e)}"}


class EarningsDataTool(Tool):
    """Tool for collecting earnings data and financial statements."""
    
    @property
    def name(self) -> str:
        return "earnings_data"
    
    @property
    def description(self) -> str:
        return "Retrieves earnings data, financial statements, and analyst estimates"
    
    def execute(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch earnings and financial data.
        
        Args:
            symbol: Stock ticker symbol
        """
        try:
            stock = yf.Ticker(symbol)
            
            # Get earnings data
            earnings = stock.earnings
            quarterly_earnings = stock.quarterly_earnings
            
            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get analyst info
            info = stock.info
            
            result = {
                "symbol": symbol,
                "earnings": {
                    "annual": earnings.to_dict() if earnings is not None else {},
                    "quarterly": quarterly_earnings.to_dict() if quarterly_earnings is not None else {}
                },
                "financial_statements": {
                    "income_statement": income_stmt.head().to_dict() if income_stmt is not None else {},
                    "balance_sheet": balance_sheet.head().to_dict() if balance_sheet is not None else {},
                    "cash_flow": cash_flow.head().to_dict() if cash_flow is not None else {}
                },
                "analyst_estimates": {
                    "target_mean_price": info.get("targetMeanPrice"),
                    "target_high_price": info.get("targetHighPrice"),
                    "target_low_price": info.get("targetLowPrice"),
                    "recommendation_mean": info.get("recommendationMean"),
                    "recommendation_key": info.get("recommendationKey"),
                    "number_of_analyst_opinions": info.get("numberOfAnalystOpinions")
                },
                "key_metrics": {
                    "revenue": info.get("totalRevenue"),
                    "gross_profit": info.get("grossProfits"),
                    "ebitda": info.get("ebitda"),
                    "net_income": info.get("netIncomeToCommon"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "profit_margins": info.get("profitMargins"),
                    "operating_margins": info.get("operatingMargins"),
                    "return_on_assets": info.get("returnOnAssets"),
                    "return_on_equity": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "book_value": info.get("bookValue"),
                    "price_to_book": info.get("priceToBook"),
                    "forward_eps": info.get("forwardEps"),
                    "trailing_eps": info.get("trailingEps")
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to fetch earnings data for {symbol}: {str(e)}"}


class MarketDataTool(Tool):
    """Tool for collecting broader market data and context."""
    
    @property
    def name(self) -> str:
        return "market_data"
    
    @property
    def description(self) -> str:
        return "Retrieves broader market data, indices, and economic indicators"
    
    def execute(self, indices: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Fetch market data for major indices and economic indicators.
        
        Args:
            indices: List of index symbols to fetch (defaults to major indices)
        """
        try:
            if indices is None:
                indices = ["^GSPC", "^DJI", "^IXIC", "^VIX", "^TNX"]  # S&P 500, Dow, NASDAQ, VIX, 10-year Treasury
            
            market_data = {}
            
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period="5d")
                    info = ticker.info
                    
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    market_data[index] = {
                        "name": info.get("longName", index),
                        "current_price": float(current_price),
                        "change": float(change),
                        "change_pct": float(change_pct),
                        "volume": int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0,
                        "high_52w": info.get("fiftyTwoWeekHigh"),
                        "low_52w": info.get("fiftyTwoWeekLow")
                    }
                    
                except Exception as e:
                    market_data[index] = {"error": f"Failed to fetch data: {str(e)}"}
                
                time.sleep(0.1)  # Rate limiting
            
            # Calculate market sentiment indicators
            sentiment_indicators = {}
            if "^VIX" in market_data and "error" not in market_data["^VIX"]:
                vix_level = market_data["^VIX"]["current_price"]
                if vix_level < 20:
                    sentiment_indicators["volatility"] = "Low (Complacent market)"
                elif vix_level < 30:
                    sentiment_indicators["volatility"] = "Moderate (Normal market)"
                else:
                    sentiment_indicators["volatility"] = "High (Fearful market)"
            
            result = {
                "indices": market_data,
                "sentiment_indicators": sentiment_indicators,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to fetch market data: {str(e)}"}


class AlphaVantageDataTool(Tool):
    """Tool for collecting data from Alpha Vantage API (if API key is available)."""
    
    @property
    def name(self) -> str:
        return "alpha_vantage_data"
    
    @property
    def description(self) -> str:
        return "Retrieves fundamental data and technical indicators from Alpha Vantage"
    
    def execute(self, symbol: str, function: str = "OVERVIEW", **kwargs) -> Dict[str, Any]:
        """
        Fetch data from Alpha Vantage API.
        
        Args:
            symbol: Stock ticker symbol
            function: Alpha Vantage function (OVERVIEW, EARNINGS, etc.)
        """
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if not api_key:
            return {"error": "Alpha Vantage API key not provided"}
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    return {"error": data["Error Message"]}
                
                if "Note" in data:
                    return {"error": "API rate limit exceeded"}
                
                result = {
                    "symbol": symbol,
                    "function": function,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                return result
            else:
                return {"error": f"HTTP error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Failed to fetch Alpha Vantage data: {str(e)}"}