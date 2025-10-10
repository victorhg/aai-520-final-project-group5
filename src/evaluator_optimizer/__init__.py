"""
Evaluator-Optimizer Module

Implements the Evaluator-Optimizer workflow pattern for iterative refinement
of investment research summaries.
"""

from .evaluator_optimizer import EvaluatorOptimizer, Feedback, State, create_mock_context

__all__ = ["EvaluatorOptimizer", "Feedback", "State", "create_mock_context"]
