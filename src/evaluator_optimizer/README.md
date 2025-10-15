# Evaluator-Optimizer Module

This module implements the **Evaluator-Optimizer workflow pattern** for iterative refinement of investment research summaries using LangGraph.

## Overview

The Evaluator-Optimizer pattern consists of two main components that work in a feedback loop:

1. **Generator**: Creates or refines investment research summaries based on context data
2. **Evaluator**: Assesses quality and provides structured feedback for improvement

The workflow continues iterating until either:
- Quality criteria are met (grade = "pass")
- Maximum iterations are reached

## Architecture

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Generator  │  ◄─────┐
└──────┬──────┘        │
       │               │
       ▼               │
┌─────────────┐        │
│  Evaluator  │        │
└──────┬──────┘        │
       │               │
       ▼               │
  Pass/Fail?           │
       │               │
       ├─ Fail ────────┘
       │
       └─ Pass
       │
       ▼
┌─────────────┐
│     END     │
└─────────────┘
```

## Quality Criteria

Summaries are evaluated against:

1. **COMPLETENESS**: Key financial metrics, sentiment, risk factors
2. **CLARITY**: Well-structured, concise, understandable
3. **ACTIONABILITY**: Clear investment recommendation with rationale
4. **EVIDENCE-BASED**: Data-backed claims from news and financials
5. **COHERENCE**: Logical flow without contradictions
6. **RISK AWARENESS**: Balanced view of opportunities and risks

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Ensure you have:
# - langchain-openai
# - langgraph
# - python-dotenv
# - pydantic
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from src.evaluator_optimizer import EvaluatorOptimizer, create_mock_context

# Initialize (loads OpenAI API key from .env)
evaluator = EvaluatorOptimizer(max_iterations=3)

# Create context data from your ingestion/processing pipeline
symbol = "AAPL"
context = {
    "financial_data": {...},
    "news_data": [...],
    "sentiment": {...}
}

# Execute the workflow
results = evaluator.execute(symbol, context)

# Access results
print(f"Final Summary: {results['final_summary']}")
print(f"Quality Score: {results['quality_score']}/10")
print(f"Passed: {results['passed']}")
print(f"Iterations: {results['iterations']}")
```

### Custom LLM Configuration

```python
from langchain_openai import ChatOpenAI

# Use custom LLM settings (e.g., different model or temperature)
custom_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.5
)

evaluator = EvaluatorOptimizer(
    llm=custom_llm,
    max_iterations=5
)

results = evaluator.execute(symbol, context)
```

## Standalone Testing

Run the module directly to test with sample context data:

```bash
# From project root
python -m src.evaluator_optimizer.evaluator_optimizer

# Or from the module directory
cd src/evaluator_optimizer
python evaluator_optimizer.py
```

**Note**: Ensure your `.env` file contains a valid `OPENAI_API_KEY` before running.

## Output Structure

The `execute()` method returns:

```python
{
    "final_summary": str,          # The optimized summary
    "quality_score": float,         # Final score (0-10)
    "iterations": int,              # Number of iterations performed
    "history": List[Dict],          # Detailed iteration history
    "passed": bool,                 # Whether quality criteria were met
    "final_grade": str,            # "pass" or "fail"
    "issues": List[str]            # Remaining issues (if any)
}
```

## Integration with Orchestrator

The module can be integrated into the main orchestrator workflow:

```python
from src.orchestrator.orchestrator import Orchestrator
from src.evaluator_optimizer import EvaluatorOptimizer

class Orchestrator:
    def __init__(self):
        self.evaluator_optimizer = EvaluatorOptimizer(max_iterations=3)
    
    def execute(self, symbol: str, instructions: str):
        # ... ingestion and processing ...
        
        # Use evaluator-optimizer for summary refinement
        results = self.evaluator_optimizer.execute(symbol, processed_context)
        
        # Store in memory
        # ...
```

## Features

- ✅ Iterative refinement with structured feedback
- ✅ Quality scoring (0-10 scale)
- ✅ Issue tracking and detailed feedback
- ✅ Iteration history for debugging
- ✅ OpenAI LLM integration (gpt-4o-mini)
- ✅ Max iteration safety limit
- ✅ Workflow visualization
- ✅ Environment-based configuration

## File Structure

```
evaluator_optimizer/
├── __init__.py                      # Package exports
├── evaluator_optimizer.py           # Main implementation
└── README.md                        # This file
```

## Example Output

```
############################################################
EVALUATOR-OPTIMIZER WORKFLOW
Symbol: AAPL
Max Iterations: 3
############################################################

============================================================
GENERATOR - Iteration 1
============================================================
Summary generated (1247 characters)

============================================================
EVALUATOR - Iteration 1
============================================================
Grade: FAIL
Quality Score: 6.5/10
Issues Found: 3
Issues: Missing risk/reward ratio, Vague entry points, No stop-loss
Feedback: Good structure but needs specific risk analysis...

🔄 Quality FAILED - Continuing to iteration 2

============================================================
GENERATOR - Iteration 2
============================================================
Summary generated (1892 characters)
Addressing feedback: Good structure but needs specific...

============================================================
EVALUATOR - Iteration 2
============================================================
Grade: PASS
Quality Score: 8.7/10
Issues Found: 0
Feedback: Excellent improvements. All criteria met.

✅ Quality PASSED - Ending optimization

############################################################
WORKFLOW COMPLETE
############################################################
Total Iterations: 2
Final Grade: PASS
Final Quality Score: 8.7/10
Quality Passed: ✅ YES
############################################################
```

## Notes

- Uses `gpt-4o-mini` by default for cost efficiency
- Feedback is structured using Pydantic models
- Suitable for production with real financial data
- Can be extended with custom evaluation criteria
- Thread-safe for parallel processing

## Related Patterns

This module implements one of three required workflow patterns:

1. **Prompt Chaining** (see: `src/processing/`)
2. **Routing** (see: `src/orchestrator/`)
3. **Evaluator-Optimizer** (this module) ✅

## References

- [LangGraph Workflows Documentation](https://langchain-ai.github.io/langgraph/tutorials/workflows/#evaluator-optimizer)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/news/building-effective-agents)
