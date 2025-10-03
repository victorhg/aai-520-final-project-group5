# Aggregator Module - Architecture & Flow

## Overview

The **Aggregator Module** is responsible for processing, normalizing, and enriching raw data from the Ingestion layer. It acts as the second stage in the data pipeline, transforming unstructured raw data into clean, structured, analysis-ready information.

## Core Principle

**Aggregation = Data Processing & Normalization**

- ✅ Clean and validate raw data
- ✅ Normalize data formats across different sources
- ✅ Calculate derived metrics and indicators
- ✅ Filter and deduplicate data
- ✅ Structure data for downstream analysis
- ✅ Enrich data with calculated features
- ❌ NO final recommendations (that's for downstream modules)
- ❌ NO sentiment analysis (handled by Processing module)
- ❌ NO summarization (handled by Summarizer module)

---

## Architecture

### Component Structure

```
src/aggregator/
├── aggregator.py             # Main coordinator
├── financial.py              # Financial data processing & metrics calculation
├── news.py                   # News data normalization & deduplication
├── memory.py                 # Memory data processing (to be added)
├── additional.py             # Additional data processing (to be added)
└── aggregator-flow.md        # This documentation
```

---

## Aggregator Responsibilities by Type

### Financial Aggregator (`financial.py`)
**Input**: Raw OHLCV data, raw company info from yfinance  
**Processing**:
- Calculate derived metrics (volatility, price changes, percentages)
- Compute volume averages
- Calculate technical indicators
- Normalize financial ratios (P/E, dividend yield, beta)
- Extract key company information
- Format historical data for analysis

**Output**: Structured financial metrics ready for analysis

### News Aggregator (`news.py`)
**Input**: Raw news articles from multiple sources (NewsAPI, Yahoo RSS, Kaggle)  
**Processing**:
- Normalize article format across sources
- Deduplicate similar/identical articles
- Filter irrelevant or low-quality articles
- Standardize date formats
- Extract and clean article text
- Add metadata (source reliability, recency score)

**Output**: Clean, deduplicated list of relevant news articles

### Memory Aggregator (`memory.py`) - To Be Implemented
**Input**: Raw past analyses and learned patterns  
**Processing**:
- Normalize historical recommendation formats
- Calculate accuracy metrics of past predictions
- Identify relevant patterns for current analysis
- Filter outdated or irrelevant insights
- Rank insights by relevance and confidence

**Output**: Structured historical insights with relevance scores

### Additional Data Aggregator (`additional.py`) - To Be Implemented
**Input**: Raw macro indicators, SEC filings, Senate trading data  
**Processing**:
- Normalize economic indicator formats
- Extract key sections from SEC filings
- Categorize Senate trading patterns
- Calculate market index correlations
- Identify significant regulatory events

**Output**: Structured contextual market data

---

## Flowcharts

### High-Level Aggregation Flow

```mermaid
graph TB
    A[Raw Data Bundle<br/>from Ingestion] --> B[Aggregator<br/>Main Coordinator]
    
    B --> C{Route by<br/>Data Type}
    
    C --> D[FinancialAggregator]
    C --> E[NewsAggregator]
    C --> F[MemoryAggregator]
    C --> G[AdditionalAggregator]
    
    D --> H[Calculate Metrics]
    D --> I[Normalize Formats]
    
    E --> J[Deduplicate Articles]
    E --> K[Filter Relevance]
    
    F --> L[Score Relevance]
    F --> M[Calculate Accuracy]
    
    G --> N[Extract Key Data]
    G --> O[Normalize Formats]
    
    H --> P[Combine All<br/>Processed Data]
    I --> P
    J --> P
    K --> P
    L --> P
    M --> P
    N --> P
    O --> P
    
    P --> Q[Aggregated Data Bundle]
    Q --> R[Processing Module<br/>Classifier/Extractor]
    
    style B fill:#4CAF50,color:#fff
    style C fill:#2196F3,color:#fff
    style P fill:#FF9800,color:#fff
    style Q fill:#9C27B0,color:#fff
```

### Detailed Financial Aggregation Flow

```mermaid
graph TB
    A[Raw Financial Data] --> B[FinancialAggregator]
    
    B --> C[Validate Data Quality]
    C --> D{Data Valid?}
    
    D -->|Yes| E[Extract Historical OHLCV]
    D -->|No| F[Log Error & Return Partial]
    
    E --> G[Calculate Price Metrics]
    G --> G1[Price Change]
    G --> G2[Price Change %]
    G --> G3[52-Week High/Low]
    
    E --> H[Calculate Volume Metrics]
    H --> H1[Avg Volume 30d]
    H --> H2[Volume Change %]
    H --> H3[Volume Trend]
    
    E --> I[Calculate Volatility]
    I --> I1[30-Day Volatility]
    I --> I2[Annualized Volatility]
    I --> I3[Volatility Percentile]
    
    E --> J[Calculate Technical Indicators]
    J --> J1[Moving Averages]
    J --> J2[RSI]
    J --> J3[MACD]
    
    K[Raw Company Info] --> L[Extract Fundamentals]
    L --> L1[P/E Ratio]
    L --> L2[Market Cap]
    L --> L3[Beta]
    L --> L4[Dividend Yield]
    
    G1 --> M[Combine All Metrics]
    G2 --> M
    G3 --> M
    H1 --> M
    H2 --> M
    H3 --> M
    I1 --> M
    I2 --> M
    I3 --> M
    J1 --> M
    J2 --> M
    J3 --> M
    L1 --> M
    L2 --> M
    L3 --> M
    L4 --> M
    
    M --> N[Structured Financial Data]
    
    style B fill:#4CAF50,color:#fff
    style M fill:#FF9800,color:#fff
    style N fill:#9C27B0,color:#fff
```

### Detailed News Aggregation Flow

```mermaid
graph TB
    A[Raw News Articles<br/>from Multiple Sources] --> B[NewsAggregator]
    
    B --> C[Normalize Article Format]
    C --> C1[Standardize Fields]
    C --> C2[Parse Dates]
    C --> C3[Clean Text]
    
    C1 --> D[Deduplicate Articles]
    C2 --> D
    C3 --> D
    
    D --> E[Similarity Detection]
    E --> E1[Title Similarity]
    E --> E2[Content Similarity]
    E --> E3[Time Window Check]
    
    E1 --> F{Duplicate?}
    E2 --> F
    E3 --> F
    
    F -->|Yes| G[Keep Highest Quality]
    F -->|No| H[Retain Article]
    
    G --> I[Filter Relevance]
    H --> I
    
    I --> J[Relevance Scoring]
    J --> J1[Keyword Match]
    J --> J2[Source Reliability]
    J --> J3[Recency Score]
    
    J1 --> K{Passes<br/>Threshold?}
    J2 --> K
    J3 --> K
    
    K -->|Yes| L[Add to Final List]
    K -->|No| M[Discard]
    
    L --> N[Rank by Relevance]
    N --> O[Clean News Dataset]
    
    style B fill:#4CAF50,color:#fff
    style D fill:#2196F3,color:#fff
    style I fill:#FF9800,color:#fff
    style O fill:#9C27B0,color:#fff
```

### Data Transformation Flow

```mermaid
graph LR
    subgraph "Raw Data (Ingestion Output)"
        A1[Unstructured]
        A2[Multiple Formats]
        A3[Duplicates]
        A4[Missing Values]
    end
    
    subgraph "Aggregator Processing"
        B1[Normalize]
        B2[Calculate]
        B3[Filter]
        B4[Enrich]
    end
    
    subgraph "Processed Data (Aggregator Output)"
        C1[Structured]
        C2[Consistent Format]
        C3[Deduplicated]
        C4[Complete & Enriched]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B3
    A4 --> B4
    
    B1 --> C1
    B1 --> C2
    B2 --> C4
    B3 --> C3
    B4 --> C4
    
    style B1 fill:#4CAF50,color:#fff
    style B2 fill:#2196F3,color:#fff
    style B3 fill:#FF9800,color:#fff
    style B4 fill:#9C27B0,color:#fff
```

### Sequence Diagram: Ingestion → Aggregator → Processing

```mermaid
sequenceDiagram
    participant I as Ingestion
    participant A as Aggregator
    participant FA as FinancialAggregator
    participant NA as NewsAggregator
    participant MA as MemoryAggregator
    participant AA as AdditionalAggregator
    participant P as Processing Module
    
    I->>A: Raw Data Bundle
    
    A->>A: Route data by type
    
    par Process Each Data Type
        A->>FA: Raw financial data
        A->>NA: Raw news articles
        A->>MA: Raw memory data
        A->>AA: Raw additional data
    end
    
    FA->>FA: Calculate metrics<br/>Normalize formats
    NA->>NA: Deduplicate<br/>Filter relevance
    MA->>MA: Score relevance<br/>Calculate accuracy
    AA->>AA: Extract key data<br/>Normalize formats
    
    FA-->>A: Processed financial data
    NA-->>A: Clean news dataset
    MA-->>A: Relevant insights
    AA-->>A: Structured context
    
    A->>A: Combine all processed data
    A-->>P: Aggregated Data Bundle
```

---

## Data Structures

### Input to Aggregator (from Ingestion)

**Raw Data Bundle Structure:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-10-03T12:00:00Z",
  "financial_data": {
    "historical": "DataFrame[OHLCV]",
    "info": { "raw company info" }
  },
  "news_data": [
    { "raw article 1" },
    { "raw article 2" }
  ],
  "memory_data": { "past analyses" },
  "additional_data": { "macro, filings, senate" },
  "errors": [],
  "status": "success"
}
```

### Output from Aggregator (Processed Data)

**Aggregated Data Bundle Structure:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-10-03T12:00:00Z",
  
  "financial_metrics": {
    "price_data": {
      "current_price": 175.50,
      "price_change": 2.30,
      "price_change_pct": 1.33,
      "52_week_high": 199.62,
      "52_week_low": 164.08
    },
    "volume_metrics": {
      "current_volume": 50234567,
      "avg_volume_30d": 48123456,
      "volume_change_pct": 4.39
    },
    "volatility_metrics": {
      "volatility_30d": 0.25,
      "annualized_volatility": 0.45,
      "volatility_percentile": 65
    },
    "fundamental_ratios": {
      "pe_ratio": 28.5,
      "market_cap": 2750000000000,
      "beta": 1.2,
      "dividend_yield": 0.0052
    },
    "technical_indicators": {
      "sma_20": 173.25,
      "sma_50": 170.80,
      "rsi_14": 58.3,
      "macd": 1.45
    },
    "company_info": {
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "summary": "Apple Inc. designs, manufactures..."
    }
  },
  
  "news_articles": [
    {
      "title": "Apple Announces Record Quarter",
      "summary": "Apple Inc. reported...",
      "source": "NewsAPI",
      "source_reliability": 0.9,
      "published": "2025-10-02T14:30:00Z",
      "recency_score": 0.95,
      "relevance_score": 0.88,
      "url": "https://..."
    }
  ],
  
  "historical_insights": {
    "past_recommendations": [
      {
        "date": "2025-09-25",
        "recommendation": "buy",
        "confidence": 0.85,
        "outcome": "correct",
        "relevance_score": 0.75
      }
    ],
    "accuracy_metrics": {
      "overall_accuracy": 0.72,
      "recent_accuracy": 0.80
    }
  },
  
  "market_context": {
    "macro_indicators": {
      "gdp_growth": 2.5,
      "inflation_rate": 3.2,
      "fed_rate": 5.25,
      "market_sentiment": "neutral"
    },
    "sector_performance": {
      "technology_sector_change": 1.2,
      "market_correlation": 0.85
    },
    "regulatory_events": [
      {
        "type": "10-Q",
        "date": "2025-09-30",
        "key_highlights": ["Revenue up 15%", "Margin expansion"]
      }
    ]
  },
  
  "data_quality": {
    "completeness": 0.95,
    "sources_processed": 4,
    "errors": []
  },
  
  "status": "success"
}
```

---

## Processing Details

### Data Normalization

**Purpose**: Ensure consistent format across different data sources

**Techniques:**
- **Date Normalization**: Convert all dates to ISO 8601 format
- **Text Cleaning**: Remove HTML tags, special characters, normalize whitespace
- **Currency Normalization**: Convert all monetary values to same currency (USD)
- **Unit Standardization**: Ensure consistent units (millions vs billions)
- **Field Mapping**: Map different field names to standard schema

### Deduplication Strategy

**News Articles:**
- **Title Similarity**: Use fuzzy matching (Levenshtein distance > 0.8)
- **Content Similarity**: Compare article text using TF-IDF cosine similarity
- **Time Window**: Articles within 24 hours on same topic
- **Quality Ranking**: Keep article from most reliable source

**Data Selection:**
- Prefer official sources over aggregators
- Prefer recent data over older duplicates
- Prefer complete records over partial

### Metric Calculation

**Financial Metrics:**
- **Volatility**: Standard deviation of returns, annualized
- **Price Changes**: Absolute and percentage changes
- **Volume Metrics**: Moving averages, volume trends
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Valuation Ratios**: P/E, P/B, PEG, dividend yield

**Quality Metrics:**
- **Data Completeness**: Percentage of non-null fields
- **Recency Score**: Time decay function for article age
- **Relevance Score**: Keyword match + source reliability
- **Confidence Score**: Based on data quality and consistency

### Filtering & Validation

**Data Quality Checks:**
- Remove records with critical missing fields
- Validate data ranges (e.g., prices > 0)
- Check for outliers and anomalies
- Verify timestamp consistency
- Ensure minimum data threshold per category

**Relevance Filtering:**
- News articles must mention the stock symbol or company name
- Filter out generic market news (unless highly relevant)
- Remove promotional or spam content
- Prioritize news from last 7 days

---

## Separation from Other Modules

### What Aggregator Does:
1. ✅ Clean and normalize raw data
2. ✅ Calculate derived metrics
3. ✅ Deduplicate and filter
4. ✅ Structure data for analysis
5. ✅ Enrich with calculated features

### What Ingestion Does (NOT Aggregator):
1. ❌ Fetch data from external APIs
2. ❌ Handle API authentication
3. ❌ Manage rate limits
4. ❌ Return raw/unprocessed data

### What Processing Does (NOT Aggregator):
1. ❌ Sentiment analysis on news
2. ❌ Entity extraction from text
3. ❌ Text classification
4. ❌ NLP preprocessing

### What Summarizer Does (NOT Aggregator):
1. ❌ Generate natural language summaries
2. ❌ Create investment narratives
3. ❌ Synthesize final recommendations

---

## Data Quality & Validation

### Quality Metrics Tracked

**Completeness Score:**
- Percentage of expected fields that are non-null
- Critical fields: symbol, price, volume, date
- Optional fields: fundamentals, technical indicators

**Consistency Score:**
- Cross-validation between sources
- Price consistency across time periods
- Volume consistency with market data

**Accuracy Score:**
- Validation against known benchmarks
- Comparison with official exchange data
- Historical accuracy of memory predictions

**Recency Score:**
- Time decay function for data age
- Higher weight for recent data
- Threshold for stale data exclusion

### Validation Rules

**Financial Data:**
- Price > 0
- Volume >= 0
- Market cap > 0
- Valid date ranges
- P/E ratio within reasonable bounds

**News Data:**
- Valid URL format
- Date within last 90 days
- Minimum text length
- Source from approved list
- Language is English

**Memory Data:**
- Valid date ranges
- Confidence scores between 0-1
- Recognized recommendation types
- Traceable to past analysis

---

## Performance Considerations

### Optimization Strategies

**Parallel Processing:**
- Process each data type independently
- Use thread pool for concurrent aggregation
- Maintain 4 concurrent workers (matching ingestion)

**Caching:**
- Cache calculated metrics for repeated queries
- Store normalized data temporarily
- Reuse technical indicator calculations

**Batching:**
- Batch process news articles for deduplication
- Batch calculate technical indicators
- Batch normalize dates and formats

**Memory Management:**
- Stream large datasets when possible
- Limit historical data retention
- Clean up intermediate results

---

## Error Handling

### Partial Failure Strategy

**Graceful Degradation:**
- If financial aggregation fails, continue with news
- Return partial results with error tracking
- Mark data quality score accordingly

**Error Categories:**
- **Data Quality Errors**: Missing fields, invalid values
- **Calculation Errors**: Division by zero, insufficient data
- **Format Errors**: Unparseable dates, malformed text

**Recovery Actions:**
- Use default values for non-critical fields
- Skip invalid records with logging
- Return available metrics with quality flags

---

## References

- **Financial Metrics**: Standard industry calculations (volatility, ratios)
- **Technical Indicators**: TA-Lib documentation
- **Text Similarity**: Scikit-learn TF-IDF, cosine similarity
- **Data Quality**: Great Expectations framework patterns

