from src.worker.base_worker import BaseWorker
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
import os
import re
from datetime import datetime, timedelta

load_dotenv()

class SufficiencyAssessment(BaseModel):
    """Structured assessment of whether memory data is sufficient to answer a query"""

    is_sufficient: bool = Field(
        description="Whether the memory data contains enough information to fully answer the query"
    )
    requires_fresh_data: bool = Field(
        description="Whether this query requires the most current data (e.g., current prices, recent news) or can use older data (e.g., follow-up questions, general analysis)"
    )
    missing_information: list[str] = Field(
        description="List of specific information gaps that would be needed to fully answer the query",
        default_factory=list
    )


class MemoryAnalyzer(BaseWorker):
    """
    Analyzes whether existing memory data is sufficient to answer a user query.
    Uses LangChain to perform intelligent assessment of data completeness.
    """

    def __init__(self):
        """Initialize the MemoryAnalyzer with OpenAI LLM"""
        super().__init__()
        self.llm = self._initialize_llm()
        self.structured_llm = self.llm.with_structured_output(SufficiencyAssessment)

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize OpenAI LLM with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for consistent analysis
            api_key=api_key
        )

    def _extract_timestamp(self, memory_snapshot: str) -> datetime:
        """
        Extract timestamp from memory snapshot.
        Looks for patterns like "Timestamp: YYYY-MM-DD HH:MM:SS" or similar.

        Returns:
            datetime object if found, None otherwise
        """
        # Common timestamp patterns
        patterns = [
            r'Timestamp:\s*(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})',
            r'Date:\s*(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})',
            r'Created:\s*(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})',
            r'Updated:\s*(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})',  # Just the timestamp
        ]

        for pattern in patterns:
            match = re.search(pattern, memory_snapshot, re.IGNORECASE)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Handle both "T" separator and space separator
                    if 'T' not in timestamp_str:
                        timestamp_str = timestamp_str.replace(' ', 'T')
                    return datetime.fromisoformat(timestamp_str)
                except ValueError:
                    continue

        return None

    def execute(self, memory_snapshot: str, query: str) -> Dict[str, Any]:
        """
        Analyze if the memory snapshot contains sufficient data to answer the query,
        considering both content relevance and data freshness.

        Args:
            memory_snapshot: String containing the current memory data
            query: User's query/question

        Returns:
            Dictionary with:
                - is_sufficient: Boolean indicating if data is sufficient
                - requires_fresh_data: Boolean indicating if query requires fresh data
                - missing_information: List of gaps
        """
        if not memory_snapshot or not memory_snapshot.strip():
            return {
                "is_sufficient": False,
                "requires_fresh_data": True,  # Conservative assumption when no data exists
                "missing_information": ["All information - no memory data exists"]
            }

        # Extract timestamp from memory snapshot
        memory_timestamp = self._extract_timestamp(memory_snapshot)
        current_dt = datetime.now()

        # Define freshness threshold: 24 hours for financial data
        freshness_threshold_hours = 24
        is_data_fresh = True
        age_hours = 0

        if memory_timestamp:
            age_hours = (current_dt - memory_timestamp).total_seconds() / 3600
            is_data_fresh = age_hours <= freshness_threshold_hours
        else:
            # If no timestamp found, assume data might be stale
            is_data_fresh = False

        # Create analysis prompt
        data_age_info = f"Data age: {age_hours:.1f} hours old" if memory_timestamp else "Data age: unknown"
        prompt = f"""You are an expert financial analyst evaluating whether existing research data is sufficient to answer a user's query.

USER QUERY:
{query}

AVAILABLE MEMORY DATA:
{memory_snapshot}

DATA FRESHNESS: {data_age_info}

ANALYSIS TASK:
Determine if the available memory data contains enough information to fully and accurately answer the user's query.

Consider:
1. Does the data directly address the query topic?
2. Are there specific facts, metrics, or analysis needed to answer the query?
3. Does this query require the most current/fresh data (e.g., current prices, breaking news, real-time metrics) or can it use older data (e.g., follow-up questions, general analysis, historical context)?
4. Are there any gaps that would require additional research?

IMPORTANT: If you are uncertain whether the data is sufficient, err on the side of caution and mark it as insufficient.

Provide a structured assessment of data sufficiency."""

        try:
            # Get structured assessment from LLM
            assessment = self.structured_llm.invoke(prompt)

            # Consider both content sufficiency and data freshness
            content_sufficient = assessment.is_sufficient
            requires_fresh = assessment.requires_fresh_data

            # Final sufficiency decision: content must be sufficient AND data must be fresh if required
            final_sufficient = content_sufficient and (not requires_fresh or is_data_fresh)

            # Build missing information list
            missing_info = assessment.missing_information.copy()
            if requires_fresh and not is_data_fresh:
                age_msg = f"Data is too old ({age_hours:.1f} hours > {freshness_threshold_hours} hour threshold)"
                if age_msg not in missing_info:
                    missing_info.insert(0, age_msg)

            # Convert to dictionary for return
            result = {
                "is_sufficient": final_sufficient,
                "requires_fresh_data": requires_fresh,
                "missing_information": missing_info
            }

            return result

        except Exception as e:
            # Fallback to basic keyword matching if LLM fails
            print(f"LLM analysis failed: {e}")
            return {
                "is_sufficient": False, # automatically mark as insufficient so data can be refetched
                "requires_fresh_data": True,  # Conservative assumption when LLM fails
                "missing_information": ["LLM analysis unavailable"]
            }


# --- Test the MemoryAnalyzer: `python -m src.memory_analyzer.memory_analyzer` ---
if __name__ == "__main__":
    # Sample memory snapshot (comprehensive AAPL analysis)
    sample_memory_snapshot = """# Investment Research Summary for Apple Inc. (AAPL)
Timestamp: 2024-06-15 14:30:00

## Company Overview
Apple Inc. is a leading technology company that designs, manufactures, and markets a variety of consumer electronics, software, and services. Its major products include the iPhone, Mac computers, iPads, wearables, and various accessories. The company is renowned for its innovation, brand loyalty, and its ecosystem of products and services.

## Financial Metrics
- **Current Price:** $247.45
- **Price Change:** $-1.89 (-0.76%)
- **Volume:** 39,218,197 shares
- **Market Capitalization:** $3.67 trillion
- **P/E Ratio:** 29.78
- **Dividend Yield:** 0.42%
- **52-Week Range:** $169.21 - $260.10
- **Beta:** 1.094 (indicating higher volatility than the market)
- **Sector:** Technology
- **Industry:** Consumer Electronics
- **30-Day Volatility:** 25.30%

## Recent Price Trend Analysis (Last 5 Days)
- **Day 1:** Close = $245.27, Volume = 61,999,100
- **Day 2:** Close = $247.66, Volume = 38,142,900
- **Day 3:** Close = $247.77, Volume = 35,478,000
- **Day 4:** Close = $249.34, Volume = 33,893,600
- **Day 5:** Close = $247.45, Volume = 39,218,197

### Summary of Price Movement
Over the last five trading days, the stock price has fluctuated slightly, showing a modest decline from $249.34 to $247.45. The volume has varied significantly, highlighting investor interest and potential volatility.

## News Sentiment Analysis
A review of recent news articles reveals several themes that may influence investor sentiment:

1. **Executive Departures:** Apple has reportedly lost another executive to Meta. This could raise concerns about leadership stability and innovation continuity.
2. **Acquisition Talks:** Reports indicate that Apple is in talks to acquire Prompt AI, suggesting potential expansion into AI technology, which could bolster its product offerings and competitive edge.
3. **Market Trends:** The broader market has experienced a downturn, with significant drops in the Dow and S&P 500, which may negatively affect AAPL's stock performance.
4. **Product Innovations:** Apple is preparing to launch its first-ever touch-screen MacBook, which could drive sales and enhance customer engagement.

## Risk Factors
- **Market Volatility:** AAPL has a beta of 1.094, indicating that it tends to be more volatile than the broader market. Recent market downturns may adversely affect the stock.
- **Executive Turnover:** Frequent departures of key personnel could signal internal challenges and affect strategic direction.
- **Regulatory Risks:** As a major player in technology, Apple faces scrutiny regarding privacy, antitrust issues, and supply chain dependencies.

## Investment Recommendation
**Recommendation: Hold**

### Rationale
1. **Valuation Metrics:** The P/E ratio of 29.78 suggests that Apple is trading at a premium relative to its earnings, which could indicate overvaluation, especially given the current market conditions.
2. **Dividend Yield:** The low dividend yield of 0.42% may not be attractive for income-focused investors, though it reflects the company's reinvestment strategy.
3. **Market Dynamics:** With ongoing market volatility and broader economic concerns, caution is warranted. Holding the stock allows investors to reassess their positions as market conditions evolve.
4. **Innovation Potential:** The potential acquisition of Prompt AI and the launch of new products could provide future growth catalysts, making it prudent to maintain a position in AAPL while monitoring developments.

## Conclusion
Apple Inc. remains a strong player in the technology sector with significant growth prospects. However, current market volatility, executive turnover, and the need for prudent evaluation of its valuation metrics suggest a cautious approach. Maintaining a "Hold" position allows for continued observation of the company's strategic moves and market performance."""

    # Initialize analyzer
    memory_analyzer = MemoryAnalyzer()

    # Get current timestamp for display purposes
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Test queries
    test_queries = [
        "What is the investment recommendation for Apple Inc.?",
        "What is the current price of AAPL?",
        "What is the dividend yield for AAPL?",
        "How volatile is Apple stock compared to the market?",
        "Can you remind me of what was my previous question?",
        "What was the price of AAPL one year ago?"
    ]

    print(f"Current timestamp: {current_timestamp}")

    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}: {query}")
        result = memory_analyzer.execute(sample_memory_snapshot, query)

        status = "SUFFICIENT" if result['is_sufficient'] else "INSUFFICIENT"
        missing = len(result['missing_information'])

        print(f"- Result: {status}")
        print(f"- Requires Fresh Data: {'YES' if result['requires_fresh_data'] else 'NO'}")
        if missing > 0:
            print(f"- Missing: {missing} items")
            for item in result['missing_information'][:2]:  # Show first 2 missing items
                print(f"  * {item}")
        print()
