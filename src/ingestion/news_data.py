"""
NLP-6: News Data Ingestion
Fetch news articles from NewsAPI and Yahoo Finance RSS
"""

from src.worker.base_worker import BaseWorker
from typing import Dict, Any, List
from datetime import datetime
import feedparser
import requests
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


class NewsDataIngestion(BaseWorker):
    """
    Fetches news data from multiple sources.
    
    Responsibilities:
    - Fetch news articles from NewsAPI
    - Fetch news from Yahoo Finance RSS feeds
    - Handle API errors and rate limits gracefully
    - Return structured list of articles
    """
    
    DEFAULT_LIMIT = 10
    
    def __init__(self):
        super().__init__()
    
    def execute(self, *inputs) -> Dict[str, Any]:
        """
        Fetch news articles for a given stock symbol.
        
        Args:
            inputs[0] (str): Stock ticker symbol (e.g., "AAPL")
            inputs[1] (int, optional): Number of articles to fetch (default: 10)
        
        Returns:
            dict: News data bundle containing:
                - articles: List of articles from all sources
                - sources_queried: Which sources were successfully queried
                - total_count: Total number of articles fetched
                - status: Success or error status
        """
        try:
            symbol = inputs[0]
            limit = inputs[1] if len(inputs) > 1 else self.DEFAULT_LIMIT
            
            articles = []
            sources_queried = []
            errors = []
            
            # Fetch from Yahoo Finance RSS (half the limit)
            try:
                yahoo_articles = self._fetch_from_yahoo_rss(symbol, limit // 2)
                articles.extend(yahoo_articles)
                sources_queried.append("yahoo_rss")
            except Exception as e:
                errors.append({"source": "yahoo_rss", "error": str(e)})
            
            # Fetch from NewsAPI (half the limit)
            if NEWS_API_KEY:
                try:
                    newsapi_articles = self._fetch_from_newsapi(symbol, limit // 2)
                    articles.extend(newsapi_articles)
                    sources_queried.append("newsapi")
                except Exception as e:
                    errors.append({"source": "newsapi", "error": str(e)})
            else:
                errors.append({"source": "newsapi", "error": "NEWS_API_KEY not found in environment"})
            
            return {
                "source": "news_aggregated",
                "symbol": symbol,
                "data": {
                    "articles": articles,
                    "sources_queried": sources_queried,
                    "total_count": len(articles)
                },
                "status": "success" if len(articles) > 0 else "partial_success",
                "error": errors if len(errors) > 0 else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "source": "news_aggregated",
                "symbol": inputs[0] if len(inputs) > 0 else "UNKNOWN",
                "data": {
                    "articles": [],
                    "sources_queried": [],
                    "total_count": 0
                },
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _fetch_from_newsapi(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch articles from NewsAPI.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{symbol} stock",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "link": article.get("url", ""),
                    "published": article.get("publishedAt", ""),
                    "summary": article.get("description", ""),
                    "source": article.get("source", {}).get("name", "NewsAPI")
                })
            return articles
        else:
            raise Exception(f"NewsAPI request failed with status {response.status_code}")
    
    def _fetch_from_yahoo_rss(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch articles from Yahoo Finance RSS feed.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
        yahoo_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(yahoo_rss)
        
        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
                "source": "Yahoo Finance"
            })
        
        return articles

