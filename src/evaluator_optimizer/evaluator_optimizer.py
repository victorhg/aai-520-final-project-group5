"""
Evaluator-Optimizer Module

Implements the Evaluator-Optimizer workflow pattern where:
1. Generator creates an investment research summary
2. Evaluator assesses quality and provides feedback
3. Loop continues until quality passes or max iterations reached
4. Final summary is stored in memory

Based on LangGraph pattern: https://langchain-ai.github.io/langgraph/tutorials/workflows/#evaluator-optimizer
"""

from typing import TypedDict, Literal, Dict, Any, List, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import os
from datetime import datetime
from dotenv import load_dotenv
from src.worker.base_worker import BaseWorker
import json

# Load environment variables from .env file
load_dotenv()

# Import OpenAI LLM from LangChain
from langchain_openai import ChatOpenAI


sample_context = {
  "symbol": "AAPL",
  "timestamp": "2025-10-10T19:47:49.137482",
  "financial_data": {
    "current_price": 245.27000427246094,
    "price_change": -8.769995727539055,
    "price_change_pct": -3.452210568232977,
    "volume": 61156139,
    "avg_volume_30d": 54861529.04545455,
    "volatility_30d": 0.2566234162744101,
    "market_cap": 3639902470144,
    "pe_ratio": 29.515041,
    "dividend_yield": 0.41,
    "52_week_high": 260.1,
    "52_week_low": 169.21,
    "beta": 1.094,
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "company_summary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod. It also provides AppleCare support and cloud services; and operates various platforms, including the App Store that allow customers to discov",
    "historical_data": [
      {
        "Open": 226.8800048828125,
        "High": 230.4499969482422,
        "Low": 226.64999389648438,
        "Close": 230.02999877929688,
        "Volume": 50208600,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 229.22000122070312,
        "High": 234.50999450683594,
        "Low": 229.02000427246094,
        "Close": 234.07000732421875,
        "Volume": 55824200,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 237.0,
        "High": 238.19000244140625,
        "Low": 235.02999877929688,
        "Close": 236.6999969482422,
        "Volume": 42699500,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 237.17999267578125,
        "High": 241.22000122070312,
        "Low": 236.32000732421875,
        "Close": 238.14999389648438,
        "Volume": 63421100,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 238.97000122070312,
        "High": 240.10000610351562,
        "Low": 237.72999572753906,
        "Close": 238.99000549316406,
        "Volume": 46508000,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 239.97000122070312,
        "High": 241.1999969482422,
        "Low": 236.64999389648438,
        "Close": 237.8800048828125,
        "Volume": 44249600,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 241.22999572753906,
        "High": 246.3000030517578,
        "Low": 240.2100067138672,
        "Close": 245.5,
        "Volume": 163741300,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 248.3000030517578,
        "High": 256.6400146484375,
        "Low": 248.1199951171875,
        "Close": 256.0799865722656,
        "Volume": 105517400,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 255.8800048828125,
        "High": 257.3399963378906,
        "Low": 253.5800018310547,
        "Close": 254.42999267578125,
        "Volume": 60275200,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 255.22000122070312,
        "High": 255.74000549316406,
        "Low": 251.0399932861328,
        "Close": 252.30999755859375,
        "Volume": 42303700,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 253.2100067138672,
        "High": 257.1700134277344,
        "Low": 251.7100067138672,
        "Close": 256.8699951171875,
        "Volume": 55202100,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 254.10000610351562,
        "High": 257.6000061035156,
        "Low": 253.77999877929688,
        "Close": 255.4600067138672,
        "Volume": 46076300,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 254.55999755859375,
        "High": 255.0,
        "Low": 253.00999450683594,
        "Close": 254.42999267578125,
        "Volume": 40127700,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 254.86000061035156,
        "High": 255.9199981689453,
        "Low": 253.11000061035156,
        "Close": 254.6300048828125,
        "Volume": 37704300,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 255.0399932861328,
        "High": 258.7900085449219,
        "Low": 254.92999267578125,
        "Close": 255.4499969482422,
        "Volume": 48713900,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 256.5799865722656,
        "High": 258.17999267578125,
        "Low": 254.14999389648438,
        "Close": 257.1300048828125,
        "Volume": 42630200,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 254.6699981689453,
        "High": 259.239990234375,
        "Low": 253.9499969482422,
        "Close": 258.0199890136719,
        "Volume": 49155600,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 257.989990234375,
        "High": 259.07000732421875,
        "Low": 255.0500030517578,
        "Close": 256.69000244140625,
        "Volume": 44664100,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 256.80999755859375,
        "High": 257.3999938964844,
        "Low": 255.42999267578125,
        "Close": 256.4800109863281,
        "Volume": 31955800,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 256.5199890136719,
        "High": 258.5199890136719,
        "Low": 256.1099853515625,
        "Close": 258.05999755859375,
        "Volume": 36496900,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 257.80999755859375,
        "High": 258.0,
        "Low": 253.13999938964844,
        "Close": 254.0399932861328,
        "Volume": 38322000,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      },
      {
        "Open": 255.0399932861328,
        "High": 256.3800048828125,
        "Low": 244.57000732421875,
        "Close": 245.27000427246094,
        "Volume": 61156139,
        "Dividends": 0.0,
        "Stock Splits": 0.0
      }
    ]
  },
  "news_data": {
    "articles": [],
    "sources_queried": [
      "yahoo_rss"
    ],
    "total_count": 0
  },
  "errors": [
    {
      "source": "news",
      "error": [
        {
          "source": "newsapi",
          "error": "NEWS_API_KEY not found in environment"
        }
      ]
    }
  ],
  "status": "success"
}


# --- State Definition ---
class State(TypedDict):
    """Graph state for the Evaluator-Optimizer workflow"""
    symbol: str  # Stock symbol being analyzed
    instructions: str  # User's request/query about the stock
    context: Dict[str, Any]  # Financial data context (news, prices, etc.)
    summary: str  # Current investment research summary
    feedback: str  # Feedback from evaluator
    grade: str  # Quality grade: "pass" or "fail"
    quality_score: float  # Numeric quality score (0-10)
    issues: List[str]  # List of identified issues
    iteration: int  # Current iteration number
    max_iterations: int  # Maximum allowed iterations
    history: List[Dict[str, Any]]  # History of iterations for tracking


# --- Structured Output Schema for Evaluation ---
class Feedback(BaseModel):
    """Structured evaluation feedback from the Evaluator"""
    
    grade: Literal["pass", "fail"] = Field(
        description="Overall quality assessment: 'pass' if summary meets quality criteria, 'fail' otherwise"
    )
    quality_score: float = Field(
        description="Numeric quality score from 0-10, where 10 is excellent",
        ge=0.0,
        le=10.0
    )
    feedback: str = Field(
        description="Detailed, actionable feedback for improving the summary if grade is 'fail'"
    )
    issues: List[str] = Field(
        description="List of specific issues identified in the summary",
        default_factory=list
    )


# --- Evaluator-Optimizer Class ---
class EvaluatorOptimizer(BaseWorker):
    """
    Implements the Evaluator-Optimizer workflow pattern for investment research summaries.
    
    The workflow:
    1. Generator creates an initial summary from context data
    2. Evaluator assesses quality against defined criteria
    3. If quality fails, provides feedback and loops back to Generator
    4. Continues until quality passes or max iterations reached
    5. Returns final optimized summary
    """
    
    # Quality criteria for investment research summaries
    QUALITY_CRITERIA = """
    A high-quality investment research summary should:
    1. COMPLETENESS: Cover key financial metrics, sentiment analysis, and risk factors
    2. CLARITY: Be well-structured, concise, and easy to understand
    3. ACTIONABILITY: Include a clear investment recommendation (buy/sell/hold) with rationale
    4. EVIDENCE-BASED: Back claims with specific data from news and financial metrics
    5. COHERENCE: Have logical flow without contradictions
    6. RISK AWARENESS: Acknowledge both opportunities and risks
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None, max_iterations: int = 3):
        """
        Initialize the Evaluator-Optimizer.
        
        Args:
            llm: Language model instance (if None, initializes OpenAI LLM from environment)
            max_iterations: Maximum refinement iterations before stopping
        """
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._initialize_openai_llm()
        
        self.max_iterations = max_iterations
        
        # Create structured output LLM for evaluation
        self.evaluator_llm = self.llm.with_structured_output(Feedback)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _initialize_openai_llm(self) -> ChatOpenAI:
        """Initialize OpenAI LLM with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
            temperature=0.7,
            api_key=api_key
        )
        return llm
    
    def _build_workflow(self) -> StateGraph:
        """Builds the LangGraph workflow for Evaluator-Optimizer pattern"""
        
        # Create the graph
        builder = StateGraph(State)
        
        # Add nodes
        builder.add_node("generator", self._generator_node)
        builder.add_node("evaluator", self._evaluator_node)
        
        # Add edges
        builder.add_edge(START, "generator")
        builder.add_edge("generator", "evaluator")
        
        # Conditional edge: loop back to generator or end
        builder.add_conditional_edges(
            "evaluator",
            self._should_continue,
            {
                "continue": "generator",  # Loop back with feedback
                "end": END  # Quality passed or max iterations reached
            }
        )
        
        # Compile the workflow
        return builder.compile()
    
    def _generator_node(self, state: State) -> Dict[str, Any]:
        """
        Generator Node: Creates or refines the investment research summary.
        
        On first iteration: creates initial summary from context
        On subsequent iterations: refines summary based on evaluator feedback
        """
        symbol = state["symbol"]
        context = state["context"]
        instructions = state.get("instructions", "")
        feedback = state.get("feedback", "")
        iteration = state.get("iteration", 0)
        
        # Format the context data
        formatted_context = self._format_context(context)
        
        # Build the prompt
        if iteration == 0:
            # Initial summary generation
            prompt = f"""You are an expert financial analyst. Create a comprehensive investment research summary for {symbol}.

USER REQUEST:
{instructions}

AVAILABLE DATA:
{formatted_context}

QUALITY REQUIREMENTS:
{self.QUALITY_CRITERIA}

INSTRUCTIONS:
1. Directly address the user's request/questions in your analysis
2. Use the provided financial data, news, and market context to support your analysis
3. If the user asked specific questions, answer them explicitly
4. If data is missing or unavailable, acknowledge it and work with what's available
5. Provide a clear investment recommendation (buy/sell/hold) with detailed rationale
6. Structure your response professionally with clear sections
7. Back all claims with specific data points from the context

Generate a well-structured investment research summary that meets all quality criteria and addresses the user's request."""
        else:
            # Refinement based on feedback
            current_summary = state["summary"]
            prompt = f"""You are refining an investment research summary for {symbol}. 

USER REQUEST:
{instructions}

CURRENT SUMMARY:
{current_summary}

EVALUATOR FEEDBACK:
{feedback}

ISSUES IDENTIFIED:
{', '.join(state.get('issues', []))}

AVAILABLE DATA:
{formatted_context}

QUALITY REQUIREMENTS:
{self.QUALITY_CRITERIA}

INSTRUCTIONS:
1. Address all issues identified by the evaluator
2. Ensure the user's original request is still fully addressed
3. Improve clarity, completeness, and actionability
4. Add missing data points or analysis where needed
5. Maintain professional structure and tone

Generate an improved version that addresses all feedback and meets quality standards."""
        
        # Generate summary
        response = self.llm.invoke(prompt)
        new_summary = response.content
        
        # Update iteration count
        new_iteration = iteration + 1
        
        print(f"\n{'='*60}")
        print(f"GENERATOR - Iteration {new_iteration}")
        print(f"{'='*60}")
        print(f"User Request: {instructions[:80]}...")
        print(f"Summary generated ({len(new_summary)} characters)")
        if iteration > 0:
            print(f"Addressing feedback: {feedback[:100]}...")
        
        return {
            "summary": new_summary,
            "iteration": new_iteration
        }
    
    def _evaluator_node(self, state: State) -> Dict[str, Any]:
        """
        Evaluator Node: Assesses the quality of the summary and provides feedback.
        
        Uses structured output to return:
        - grade: "pass" or "fail"
        - quality_score: 0-10
        - feedback: actionable improvement suggestions
        - issues: specific problems identified
        """
        summary = state["summary"]
        symbol = state["symbol"]
        instructions = state.get("instructions", "")
        iteration = state["iteration"]
        
        # Evaluation prompt
        prompt = f"""You are a senior financial analyst evaluating an investment research summary for {symbol}.

USER'S ORIGINAL REQUEST:
{instructions}

SUMMARY TO EVALUATE:
{summary}

EVALUATION CRITERIA:
{self.QUALITY_CRITERIA}

ASSESSMENT REQUIREMENTS:
1. Does the summary directly address the user's request/questions?
2. Is the analysis backed by specific data points?
3. Are all quality criteria met (completeness, clarity, actionability, evidence-based, coherence, risk awareness)?
4. Is the investment recommendation clear and well-justified?
5. Are there any contradictions or unsupported claims?
6. Is the structure professional and easy to follow?

Provide:
- grade: "pass" if the summary meets professional standards and addresses the user's request, "fail" if significant improvements needed
- quality_score: 0-10 (be generous with 7+ for good work, reserve 9+ for exceptional analysis)
- feedback: Specific, actionable suggestions for improvement (if grade is "fail")
- issues: List specific problems (missing data, unclear reasoning, unanswered questions, etc.)

Be fair but thorough. A passing grade means the summary is publication-ready and fully addresses the user's needs."""
        
        # Get structured evaluation
        evaluation = self.evaluator_llm.invoke(prompt)
        
        # Track iteration history
        history = state.get("history", [])
        history.append({
            "iteration": iteration,
            "summary_length": len(summary),
            "grade": evaluation.grade,
            "quality_score": evaluation.quality_score,
            "issues_count": len(evaluation.issues)
        })
        
        print(f"\n{'='*60}")
        print(f"EVALUATOR - Iteration {iteration}")
        print(f"{'='*60}")
        print(f"Grade: {evaluation.grade.upper()}")
        print(f"Quality Score: {evaluation.quality_score}/10")
        print(f"Issues Found: {len(evaluation.issues)}")
        if evaluation.issues:
            for i, issue in enumerate(evaluation.issues, 1):
                print(f"  {i}. {issue}")
        print(f"Feedback: {evaluation.feedback[:150]}...")
        
        return {
            "grade": evaluation.grade,
            "quality_score": evaluation.quality_score,
            "feedback": evaluation.feedback,
            "issues": evaluation.issues,
            "history": history
        }
    
    def _should_continue(self, state: State) -> Literal["continue", "end"]:
        """
        Conditional routing: decide whether to continue refinement or end.
        
        Continue if:
        - Grade is "fail" AND
        - Haven't reached max iterations
        
        End if:
        - Grade is "pass" OR
        - Max iterations reached
        """
        grade = state["grade"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        if grade == "pass":
            print(f"\n✅ Quality PASSED - Ending optimization")
            return "end"
        elif iteration >= max_iterations:
            print(f"\n⚠️  Max iterations ({max_iterations}) reached - Ending optimization")
            return "end"
        else:
            print(f"\n🔄 Quality FAILED - Continuing to iteration {iteration + 1}")
            return "continue"
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Formats context data into a readable string for prompts"""
        formatted = []
        
        # Format financial data with key metrics
        if "financial_data" in context:
            fin_data = context["financial_data"]
            formatted.append("=== FINANCIAL METRICS ===")
            formatted.append(f"Symbol: {fin_data.get('symbol', 'N/A')}")
            formatted.append(f"Current Price: ${fin_data.get('current_price', 0):.2f}")
            formatted.append(f"Price Change: ${fin_data.get('price_change', 0):.2f} ({fin_data.get('price_change_pct', 0):.2f}%)")
            formatted.append(f"Volume: {fin_data.get('volume', 0):,}")
            formatted.append(f"Market Cap: ${fin_data.get('market_cap', 0):,}")
            formatted.append(f"P/E Ratio: {fin_data.get('pe_ratio', 0):.2f}")
            formatted.append(f"Dividend Yield: {fin_data.get('dividend_yield', 0):.2f}%")
            formatted.append(f"52-Week Range: ${fin_data.get('52_week_low', 0):.2f} - ${fin_data.get('52_week_high', 0):.2f}")
            formatted.append(f"Beta: {fin_data.get('beta', 0):.3f}")
            formatted.append(f"Sector: {fin_data.get('sector', 'N/A')}")
            formatted.append(f"Industry: {fin_data.get('industry', 'N/A')}")
            
            # Add volatility if available
            if 'volatility_30d' in fin_data:
                formatted.append(f"30-Day Volatility: {fin_data['volatility_30d']:.2%}")
            
            # Add company summary if available
            if 'company_summary' in fin_data and fin_data['company_summary']:
                formatted.append(f"\nCompany Overview: {fin_data['company_summary'][:300]}...")
            
            # Add recent price trends from historical data
            if 'historical_data' in fin_data and fin_data['historical_data']:
                hist = fin_data['historical_data']
                if len(hist) >= 5:
                    formatted.append("\nRecent Price Trend (Last 5 Days):")
                    for i, day in enumerate(hist[-5:], 1):
                        formatted.append(f"  Day {i}: Close=${day.get('Close', 0):.2f}, Volume={day.get('Volume', 0):,}")
        
        # Format news data
        if "news_data" in context:
            news = context["news_data"]
            if isinstance(news, dict) and "articles" in news:
                articles = news["articles"]
                total = news.get("total_count", 0)
                formatted.append(f"\n=== NEWS ANALYSIS ===")
                formatted.append(f"Total Articles Found: {total}")
                
                if articles and len(articles) > 0:
                    formatted.append("\nRecent Headlines:")
                    for i, article in enumerate(articles[:5], 1):
                        title = article.get('title', 'No title')
                        date = article.get('publishedAt', article.get('date', 'N/A'))
                        formatted.append(f"  {i}. {title} ({date})")
                else:
                    formatted.append("No recent news articles available.")
            else:
                formatted.append("\n=== NEWS ANALYSIS ===")
                formatted.append("News data format not recognized or unavailable.")
        
        # Format sentiment if available
        if "sentiment" in context:
            formatted.append(f"\n=== SENTIMENT ANALYSIS ===")
            formatted.append(f"Overall Sentiment: {context['sentiment']}")
        
        # Add any errors encountered
        if "errors" in context and context["errors"]:
            formatted.append(f"\n=== DATA COLLECTION NOTES ===")
            for error in context["errors"]:
                source = error.get("source", "unknown")
                error_detail = error.get("error", "Unknown error")
                formatted.append(f"Note: {source} - {error_detail}")
        
        # Add timestamp
        if "timestamp" in context:
            formatted.append(f"\nData Retrieved: {context['timestamp']}")
        
        return "\n".join(formatted) if formatted else "No context data available"
    
    def execute(self, symbol: str, context: Dict[str, Any], instructions: str) -> Dict[str, Any]:
        """
        Execute the Evaluator-Optimizer workflow.
        
        Args:
            symbol: Stock symbol to analyze
            context: Dictionary containing financial data, news, sentiment, etc.
            instructions: User's request/query (e.g., "Should I buy AAPL?" or "Analyze the potential for AAPL stock")
        
        Returns:
            Dictionary with:
                - final_summary: The optimized summary
                - quality_score: Final quality score
                - iterations: Number of iterations performed
                - history: Detailed iteration history
                - passed: Whether quality criteria were met
        """
        print(f"\n{'#'*60}")
        print(f"EVALUATOR-OPTIMIZER WORKFLOW")
        print(f"Symbol: {symbol}")
        print(f"User Request: {instructions}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"{'#'*60}")
        
        # Initialize state
        initial_state = {
            "symbol": symbol,
            "instructions": instructions,
            "context": context,
            "summary": "",
            "feedback": "",
            "grade": "",
            "quality_score": 0.0,
            "issues": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "history": []
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Prepare results
        results = {
            "final_summary": final_state["summary"],
            "quality_score": final_state["quality_score"],
            "iterations": final_state["iteration"],
            "history": final_state["history"],
            "passed": final_state["grade"] == "pass",
            "final_grade": final_state["grade"],
            "issues": final_state["issues"]
        }
        
        print(f"\n{'#'*60}")
        print(f"WORKFLOW COMPLETE")
        print(f"{'#'*60}")
        print(f"Total Iterations: {results['iterations']}")
        print(f"Final Grade: {results['final_grade'].upper()}")
        print(f"Final Quality Score: {results['quality_score']}/10")
        print(f"Quality Passed: {'✅ YES' if results['passed'] else '❌ NO'}")
        print(f"{'#'*60}\n")
        
        return results
    
    def visualize(self, output_path: str = "evaluator_optimizer_graph.png"):
        """
        Visualize the workflow graph.
        
        Args:
            output_path: Path to save the graph image
        """
        try:
            img_data = self.workflow.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(img_data)
            print(f"Workflow graph saved to: {output_path}")
            return Image(img_data)
        except Exception as e:
            print(f"Could not generate graph visualization: {e}")
            return None


# --- Mock Data for Testing ---
def create_mock_context(symbol: str) -> Dict[str, Any]:
    """Creates mock context data for testing"""
    return {
        "financial_data": {
            "symbol": symbol,
            "current_price": 178.50,
            "price_change": 2.3,
            "price_change_percent": 1.31,
            "volume": 45_230_000,
            "market_cap": "2.8T",
            "pe_ratio": 29.5,
            "52_week_high": 199.62,
            "52_week_low": 164.08
        },
        "news_data": [
            {
                "title": f"{symbol} announces new product line with AI integration",
                "sentiment": "positive",
                "date": "2025-10-06"
            },
            {
                "title": f"Analysts raise price target for {symbol}",
                "sentiment": "positive",
                "date": "2025-10-05"
            },
            {
                "title": f"{symbol} faces regulatory scrutiny in EU",
                "sentiment": "negative",
                "date": "2025-10-04"
            }
        ],
        "sentiment": {
            "overall": "positive",
            "score": 0.65,
            "confidence": 0.82
        },
        "analysis_date": datetime.now().isoformat()
    }


# --- Main Execution for Testing ---
if __name__ == "__main__":
    print("="*60)
    print("EVALUATOR-OPTIMIZER MODULE - STANDALONE TEST")
    print("="*60)
    
    # Initialize Evaluator-Optimizer
    print("\n1. Initializing Evaluator-Optimizer...")
    try:
        evaluator_optimizer = EvaluatorOptimizer(max_iterations=3)
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # Get context data
    symbol = "AAPL"
    # Use real ingestion data (lazy import to avoid numpy compatibility issues)
    print("\n2. Fetching real market data from Ingestion...")
    from src.ingestion.ingestion import Ingestion
    
    ingestion_worker = Ingestion()
    context = ingestion_worker.execute(symbol)
    print(f"✅ Real data retrieved:")
    print(f"   - Symbol: {context.get('symbol', 'N/A')}")
    print(f"   - Status: {context.get('status', 'N/A')}")
    print(f"   - Financial Data: {'✓' if 'financial_data' in context else '✗'}")
    print(f"   - News Data: {'✓' if 'news_data' in context else '✗'}")
    
    # Test with user instructions
    instructions = "Analyze the potential for AAPL stock over the past month and provide a buy/sell/hold recommendation with detailed rationale."
    
    print(f"\n3. Running Evaluator-Optimizer Workflow...")
    print(f"   User Query: {instructions}")
    
    # Execute workflow
    results = evaluator_optimizer.execute(symbol, context, instructions)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n📊 INVESTMENT RESEARCH SUMMARY:")
    print("-" * 60)
    print(results['final_summary'])
    print("-" * 60)
    
    print(f"\n📈 METRICS:")
    print(f"  - Iterations: {results['iterations']}")
    print(f"  - Quality Score: {results['quality_score']}/10")
    print(f"  - Quality Check: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
    print(f"  - Final Grade: {results['final_grade'].upper()}")
    
    if results['issues']:
        print(f"\n⚠️  REMAINING ISSUES:")
        for i, issue in enumerate(results['issues'], 1):
            print(f"  {i}. {issue}")
    
    print(f"\n📝 ITERATION HISTORY:")
    for record in results['history']:
        status = "✅" if record['grade'] == 'pass' else "❌"
        print(f"  {status} Iteration {record['iteration']}: "
              f"Score={record['quality_score']:.1f}/10, "
              f"Length={record['summary_length']} chars, "
              f"Issues={record['issues_count']}")
    
    # Visualize workflow
    print("\n" + "="*60)
    print("Generating workflow visualization...")
    evaluator_optimizer.visualize()
    
    print("\n✅ Test complete!")
