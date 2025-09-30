from ..worker.base_worker import BaseWorker
import feedparser
import os
import yfinance as yf
import requests

from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


class NewsAggregator(BaseWorker):
    LIMIT_ARTICLES = 5
    def __init__(self):
        self.limit = 5

        super().__init__()
    def execute(self, *inputs) -> str:
        """
            Fetch latest news articles for a given stock ticker.
            
            Args:
                symbol: Stock ticker symbol
                limit: Number of articles to fetch
            Returns:
                A list of news articles with title, summary, link, published date, and source.
        """
        symbol = inputs[0]
        limit = inputs[1] if len(inputs) > 1 else self.LIMIT_ARTICLES

        # SCRAPING YAHOO FINANCE RSS FEED
        yahoo_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
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
            "q": f"{symbol} stock",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": self.limit//2,
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