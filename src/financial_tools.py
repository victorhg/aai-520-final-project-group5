
from datetime import datetime, timedelta
import feedparser
import os
import yfinance as yf
import requests

from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


class NewsTools:
    def __init__(self):
        pass

    def get_latest_news(ticker, limit=5):              
        """
            Fetch latest news articles for a given stock ticker.
            
            Args:
                ticker: Stock ticker symbol
                limit: Number of articles to fetch
            Returns:
                A list of news articles with title, summary, link, published date, and source.
        """

        # SCRAPING YAHOO FINANCE RSS FEED
        yahoo_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        yahoo_feed = feedparser.parse(yahoo_rss)

        news = []
        for entry in yahoo_feed.entries[:limit//2]:
            news.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
                "source": "Yahoo Finance"
        })
        
        # SCRAPING NEWSAPI
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{ticker} stock",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit//2,
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for article in data.get("articles", []):
                news.append({
                    "title": article["title"],
                    "link": article["url"],
                    "published": article["publishedAt"],
                    "summary": article.get("description", ""),
                    "source": article["source"]["name"]
                })
        
        return news





class StockTools:
    def __init__(self):
        self.stock = None

    def get_stock_info( symbol, period: str = "1y"):
        """
            Fetch stock data using yfinance.
            
            Args:
                symbol: Stock ticker symbol
                period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            Returns:
                A dictionary with stock metrics and historical data.
        """
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
    