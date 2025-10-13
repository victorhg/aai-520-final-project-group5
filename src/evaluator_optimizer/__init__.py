"""
Evaluator-Optimizer Module

Implements the Evaluator-Optimizer workflow pattern for iterative refinement
of investment research summaries.
"""

from .evaluator_optimizer import EvaluatorOptimizer, Feedback, State

__all__ = ["EvaluatorOptimizer", "Feedback", "State"]
