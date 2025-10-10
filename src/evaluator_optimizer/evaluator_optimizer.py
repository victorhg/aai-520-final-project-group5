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

# Load environment variables from .env file
load_dotenv()

# Import OpenAI LLM from LangChain
from langchain_openai import ChatOpenAI


# --- State Definition ---
class State(TypedDict):
    """Graph state for the Evaluator-Optimizer workflow"""
    symbol: str  # Stock symbol being analyzed
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
        feedback = state.get("feedback", "")
        iteration = state.get("iteration", 0)
        
        # Build the prompt
        if iteration == 0:
            # Initial summary generation
            prompt = f"""Create a comprehensive investment research summary for {symbol}.

Context Data:
{self._format_context(context)}

{self.QUALITY_CRITERIA}

Generate a well-structured investment research summary that meets all quality criteria."""
        else:
            # Refinement based on feedback
            current_summary = state["summary"]
            prompt = f"""Refine the following investment research summary for {symbol} based on the feedback provided.

Current Summary:
{current_summary}

Evaluator Feedback:
{feedback}

Issues Identified:
{', '.join(state.get('issues', []))}

{self.QUALITY_CRITERIA}

Generate an improved version that addresses all feedback and issues."""
        
        # Generate summary
        response = self.llm.invoke(prompt)
        new_summary = response.content
        
        # Update iteration count
        new_iteration = iteration + 1
        
        print(f"\n{'='*60}")
        print(f"GENERATOR - Iteration {new_iteration}")
        print(f"{'='*60}")
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
        iteration = state["iteration"]
        
        # Evaluation prompt
        prompt = f"""Evaluate the quality of this investment research summary for {symbol}.

Summary to Evaluate:
{summary}

{self.QUALITY_CRITERIA}

Assess the summary against these criteria and provide:
1. A grade: "pass" if it meets quality standards, "fail" if it needs improvement
2. A quality score from 0-10
3. Specific, actionable feedback for improvement (if grade is "fail")
4. List of issues found

Be critical and thorough. A passing grade requires excellence in all criteria."""
        
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
            print(f"Issues: {', '.join(evaluation.issues)}")
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
        
        if "financial_data" in context:
            formatted.append("Financial Data:")
            formatted.append(str(context["financial_data"])[:500])
        
        if "news_data" in context:
            formatted.append("\nNews Data:")
            formatted.append(str(context["news_data"])[:500])
        
        if "sentiment" in context:
            formatted.append(f"\nSentiment: {context['sentiment']}")
        
        return "\n".join(formatted) if formatted else "No context data available"
    
    def execute(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Evaluator-Optimizer workflow.
        
        Args:
            symbol: Stock symbol to analyze
            context: Dictionary containing financial data, news, sentiment, etc.
        
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
        print(f"Max Iterations: {self.max_iterations}")
        print(f"{'#'*60}")
        
        # Initialize state
        initial_state = {
            "symbol": symbol,
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
    try:
        evaluator_optimizer = EvaluatorOptimizer(max_iterations=3)
    except ValueError as e:
        exit(1)
    
    # Test with mock context data
    symbol = "AAPL"
    context = create_mock_context(symbol)
    results = evaluator_optimizer.execute(symbol, context)
    
    print(f"\nFinal Summary:")
    print(f"{results['final_summary']}\n")
    print(f"Metrics:")
    print(f"  - Iterations: {results['iterations']}")
    print(f"  - Quality Score: {results['quality_score']}/10")
    print(f"  - Passed: {results['passed']}")
    print(f"  - Final Grade: {results['final_grade']}")
    
    if results['issues']:
        print(f"\nRemaining Issues:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    for record in results['history']:
        print(f"  Iteration {record['iteration']}: "
              f"Score={record['quality_score']}/10, "
              f"Grade={record['grade']}, "
              f"Issues={record['issues_count']}")
    
    evaluator_optimizer.visualize()
