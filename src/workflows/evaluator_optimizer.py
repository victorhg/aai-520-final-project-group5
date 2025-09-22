"""
Evaluator-Optimizer Workflow: Generate → Evaluate → Refine

This workflow demonstrates iterative improvement of investment analysis through
quality evaluation and feedback-driven refinement to optimize output quality.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import openai


class EvaluationDimension(Enum):
    """Dimensions for evaluating analysis quality."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    ACTIONABILITY = "actionability"
    RISK_ASSESSMENT = "risk_assessment"
    DATA_QUALITY = "data_quality"
    LOGICAL_CONSISTENCY = "logical_consistency"
    INVESTMENT_INSIGHT = "investment_insight"


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating analysis quality."""
    dimension: EvaluationDimension
    weight: float  # 0-1, relative importance
    min_threshold: float  # Minimum acceptable score
    description: str


@dataclass
class EvaluationScore:
    """Score for a specific evaluation dimension."""
    dimension: EvaluationDimension
    score: float  # 0-10 scale
    reasoning: str
    improvement_suggestions: List[str]


@dataclass
class AnalysisEvaluation:
    """Complete evaluation of an analysis."""
    overall_score: float
    dimension_scores: Dict[EvaluationDimension, EvaluationScore]
    strengths: List[str]
    weaknesses: List[str]
    improvement_recommendations: List[str]
    meets_threshold: bool
    evaluation_metadata: Dict[str, Any]


@dataclass
class OptimizationIteration:
    """Represents one iteration of the optimization process."""
    iteration: int
    analysis: Dict[str, Any]
    evaluation: AnalysisEvaluation
    refinement_strategy: str
    improvement_delta: float


class AnalysisGenerator:
    """Generates initial investment analysis that will be iteratively improved."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def generate_initial_analysis(self, financial_data: Dict[str, Any], symbol: str, research_goal: str) -> Dict[str, Any]:
        """Generate initial investment analysis from financial data."""
        
        prompt = f"""
        Generate a comprehensive investment analysis for {symbol} based on the provided financial data:
        
        Research Goal: {research_goal}
        
        Financial Data:
        {json.dumps(financial_data, indent=2, default=str)[:3000]}...
        
        Provide a structured investment analysis in JSON format:
        {{
            "executive_summary": {{
                "investment_thesis": "Clear 2-3 sentence investment thesis",
                "recommendation": "buy|hold|sell",
                "confidence_level": 8.5,
                "target_price": 145.50,
                "time_horizon": "6-12 months"
            }},
            "fundamental_analysis": {{
                "valuation_assessment": "undervalued|fairly_valued|overvalued",
                "financial_health": "strong|moderate|weak", 
                "growth_prospects": "high|medium|low",
                "competitive_position": "leader|competitor|follower",
                "key_financial_metrics": {{
                    "pe_ratio": 15.2,
                    "revenue_growth": "12.5%",
                    "profit_margin": "18.3%",
                    "debt_ratio": 0.35,
                    "roe": "22.1%"
                }}
            }},
            "market_analysis": {{
                "technical_outlook": "bullish|neutral|bearish",
                "price_momentum": "strong|moderate|weak",
                "volume_analysis": "healthy|average|concerning",
                "support_resistance": {{"support": 120.50, "resistance": 140.25}},
                "market_sentiment": "positive|neutral|negative"
            }},
            "risk_assessment": {{
                "overall_risk": "low|medium|high",
                "key_risks": ["risk1", "risk2", "risk3"],
                "risk_mitigation": ["mitigation1", "mitigation2"],
                "downside_scenarios": ["scenario1", "scenario2"],
                "volatility_assessment": "low|medium|high"
            }},
            "catalyst_analysis": {{
                "positive_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
                "negative_catalysts": ["catalyst1", "catalyst2"],
                "upcoming_events": ["event1", "event2"],
                "timeline_expectations": "short_term|medium_term|long_term"
            }},
            "investment_strategy": {{
                "entry_strategy": "immediate|gradual|wait_for_pullback",
                "position_sizing": "full|half|small",
                "exit_strategy": "profit_target|stop_loss|time_based",
                "monitoring_points": ["point1", "point2", "point3"]
            }},
            "supporting_evidence": {{
                "data_sources": ["source1", "source2", "source3"],
                "key_data_points": ["point1", "point2", "point3"],
                "analytical_assumptions": ["assumption1", "assumption2"],
                "confidence_factors": ["factor1", "factor2"]
            }}
        }}
        
        Ensure the analysis is comprehensive, well-reasoned, and actionable.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)
            
            # Add generation metadata
            analysis["generation_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "model": self.model,
                "symbol": symbol,
                "research_goal": research_goal,
                "version": "initial"
            }
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Failed to generate initial analysis: {str(e)}",
                "symbol": symbol,
                "research_goal": research_goal
            }


class AnalysisEvaluator:
    """Evaluates the quality of investment analysis across multiple dimensions."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.evaluation_criteria = self._setup_evaluation_criteria()
    
    def _setup_evaluation_criteria(self) -> List[EvaluationCriteria]:
        """Setup standard evaluation criteria for investment analysis."""
        return [
            EvaluationCriteria(
                dimension=EvaluationDimension.ACCURACY,
                weight=0.20,
                min_threshold=7.0,
                description="Factual correctness and data accuracy"
            ),
            EvaluationCriteria(
                dimension=EvaluationDimension.COMPLETENESS,
                weight=0.15,
                min_threshold=7.0,
                description="Comprehensive coverage of relevant factors"
            ),
            EvaluationCriteria(
                dimension=EvaluationDimension.RELEVANCE,
                weight=0.15,
                min_threshold=6.5,
                description="Relevance to investment decision-making"
            ),
            EvaluationCriteria(
                dimension=EvaluationDimension.ACTIONABILITY,
                weight=0.20,
                min_threshold=7.5,
                description="Clear, specific, and actionable recommendations"
            ),
            EvaluationCriteria(
                dimension=EvaluationDimension.RISK_ASSESSMENT,
                weight=0.15,
                min_threshold=7.0,
                description="Thorough identification and assessment of risks"
            ),
            EvaluationCriteria(
                dimension=EvaluationDimension.LOGICAL_CONSISTENCY,
                weight=0.10,
                min_threshold=8.0,
                description="Internal logical consistency and coherent reasoning"
            ),
            EvaluationCriteria(
                dimension=EvaluationDimension.INVESTMENT_INSIGHT,
                weight=0.05,
                min_threshold=6.0,
                description="Novel insights and value-added perspective"
            )
        ]
    
    def evaluate_analysis(self, analysis: Dict[str, Any], symbol: str, financial_data: Dict[str, Any]) -> AnalysisEvaluation:
        """Comprehensively evaluate the quality of an investment analysis."""
        
        # Evaluate each dimension
        dimension_scores = {}
        for criteria in self.evaluation_criteria:
            score = self._evaluate_dimension(analysis, criteria, symbol, financial_data)
            dimension_scores[criteria.dimension] = score
        
        # Calculate weighted overall score
        overall_score = sum(
            score.score * criteria.weight 
            for criteria in self.evaluation_criteria 
            for score in [dimension_scores[criteria.dimension]]
        )
        
        # Check if meets minimum thresholds
        meets_threshold = all(
            dimension_scores[criteria.dimension].score >= criteria.min_threshold
            for criteria in self.evaluation_criteria
        )
        
        # Generate overall strengths and weaknesses
        strengths, weaknesses, recommendations = self._synthesize_evaluation_feedback(
            dimension_scores, analysis, symbol
        )
        
        return AnalysisEvaluation(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_recommendations=recommendations,
            meets_threshold=meets_threshold,
            evaluation_metadata={
                "evaluated_at": datetime.now().isoformat(),
                "evaluator_model": self.model,
                "symbol": symbol,
                "criteria_count": len(self.evaluation_criteria)
            }
        )
    
    def _evaluate_dimension(self, analysis: Dict[str, Any], criteria: EvaluationCriteria, symbol: str, financial_data: Dict[str, Any]) -> EvaluationScore:
        """Evaluate a specific dimension of the analysis."""
        
        dimension_prompts = {
            EvaluationDimension.ACCURACY: f"""
            Evaluate the accuracy of this {symbol} investment analysis:
            {json.dumps(analysis, indent=2)[:2000]}...
            
            Rate accuracy (1-10) considering:
            1. Factual correctness of data points
            2. Proper use of financial metrics
            3. Realistic price targets and forecasts
            4. Accuracy of market characterizations
            
            Provide JSON response:
            {{
                "score": 8.5,
                "reasoning": "Detailed explanation of accuracy assessment",
                "improvement_suggestions": ["suggestion1", "suggestion2"]
            }}
            """,
            
            EvaluationDimension.COMPLETENESS: f"""
            Evaluate the completeness of this {symbol} investment analysis:
            {json.dumps(analysis, indent=2)[:2000]}...
            
            Rate completeness (1-10) considering:
            1. Coverage of fundamental factors
            2. Technical analysis inclusion
            3. Risk factor identification
            4. Market context consideration
            5. Catalyst identification
            
            Provide JSON response with score, reasoning, and improvement suggestions.
            """,
            
            EvaluationDimension.ACTIONABILITY: f"""
            Evaluate how actionable this {symbol} investment analysis is:
            {json.dumps(analysis, indent=2)[:2000]}...
            
            Rate actionability (1-10) considering:
            1. Clarity of investment recommendation
            2. Specific entry/exit strategies
            3. Position sizing guidance
            4. Monitoring instructions
            5. Timeline specificity
            
            Provide JSON response with score, reasoning, and improvement suggestions.
            """,
            
            EvaluationDimension.RISK_ASSESSMENT: f"""
            Evaluate the risk assessment quality in this {symbol} analysis:
            {json.dumps(analysis, indent=2)[:2000]}...
            
            Rate risk assessment (1-10) considering:
            1. Identification of key risks
            2. Risk quantification attempts
            3. Downside scenario planning
            4. Risk mitigation strategies
            5. Volatility considerations
            
            Provide JSON response with score, reasoning, and improvement suggestions.
            """,
            
            EvaluationDimension.LOGICAL_CONSISTENCY: f"""
            Evaluate the logical consistency of this {symbol} analysis:
            {json.dumps(analysis, indent=2)[:2000]}...
            
            Rate logical consistency (1-10) considering:
            1. Internal coherence of arguments
            2. Consistency between different sections
            3. Logical flow of reasoning
            4. Absence of contradictions
            
            Provide JSON response with score, reasoning, and improvement suggestions.
            """,
            
            EvaluationDimension.INVESTMENT_INSIGHT: f"""
            Evaluate the investment insight quality in this {symbol} analysis:
            {json.dumps(analysis, indent=2)[:2000]}...
            
            Rate investment insight (1-10) considering:
            1. Novel perspectives or insights
            2. Value-added analysis beyond obvious
            3. Unique angle or differentiated view
            4. Depth of market understanding
            
            Provide JSON response with score, reasoning, and improvement suggestions.
            """
        }
        
        # Default prompt for other dimensions
        if criteria.dimension not in dimension_prompts:
            prompt = f"""
            Evaluate the {criteria.dimension.value} of this investment analysis for {symbol}.
            Analysis: {json.dumps(analysis, indent=2)[:1500]}...
            
            Rate on 1-10 scale and provide reasoning and improvement suggestions in JSON format.
            """
        else:
            prompt = dimension_prompts[criteria.dimension]
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            evaluation_text = response.choices[0].message.content
            evaluation_data = json.loads(evaluation_text)
            
            return EvaluationScore(
                dimension=criteria.dimension,
                score=evaluation_data.get("score", 5.0),
                reasoning=evaluation_data.get("reasoning", "No reasoning provided"),
                improvement_suggestions=evaluation_data.get("improvement_suggestions", [])
            )
            
        except Exception as e:
            return EvaluationScore(
                dimension=criteria.dimension,
                score=5.0,
                reasoning=f"Evaluation failed: {str(e)}",
                improvement_suggestions=["Re-evaluate this dimension manually"]
            )
    
    def _synthesize_evaluation_feedback(self, dimension_scores: Dict[EvaluationDimension, EvaluationScore], analysis: Dict[str, Any], symbol: str) -> Tuple[List[str], List[str], List[str]]:
        """Synthesize overall strengths, weaknesses, and recommendations."""
        
        # Extract strengths (high-scoring dimensions)
        strengths = []
        weaknesses = []
        recommendations = []
        
        for dimension, score in dimension_scores.items():
            if score.score >= 8.0:
                strengths.append(f"Strong {dimension.value}: {score.reasoning[:100]}...")
            elif score.score < 6.5:
                weaknesses.append(f"Weak {dimension.value}: {score.reasoning[:100]}...")
            
            recommendations.extend(score.improvement_suggestions)
        
        # Generate overall synthesis
        synthesis_prompt = f"""
        Synthesize evaluation feedback for this {symbol} investment analysis:
        
        Dimension Scores:
        {json.dumps({dim.value: score.score for dim, score in dimension_scores.items()}, indent=2)}
        
        Provide top 3 overall strengths, top 3 weaknesses, and top 5 improvement recommendations
        in JSON format:
        {{
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"], 
            "recommendations": ["rec1", "rec2", "rec3", "rec4", "rec5"]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3
            )
            
            synthesis_data = json.loads(response.choices[0].message.content)
            return (
                synthesis_data.get("strengths", strengths[:3]),
                synthesis_data.get("weaknesses", weaknesses[:3]),
                synthesis_data.get("recommendations", recommendations[:5])
            )
            
        except:
            return strengths[:3], weaknesses[:3], recommendations[:5]


class AnalysisOptimizer:
    """Optimizes investment analysis based on evaluation feedback."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def optimize_analysis(self, analysis: Dict[str, Any], evaluation: AnalysisEvaluation, symbol: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize analysis based on evaluation feedback."""
        
        # Determine optimization strategy
        optimization_strategy = self._determine_optimization_strategy(evaluation)
        
        # Apply optimizations
        optimized_analysis = self._apply_optimizations(
            analysis, evaluation, optimization_strategy, symbol, financial_data
        )
        
        return optimized_analysis
    
    def _determine_optimization_strategy(self, evaluation: AnalysisEvaluation) -> str:
        """Determine the best optimization strategy based on evaluation."""
        
        # Find the lowest scoring dimensions
        weak_dimensions = [
            dim.value for dim, score in evaluation.dimension_scores.items()
            if score.score < 6.5
        ]
        
        if not weak_dimensions:
            return "minor_refinement"
        elif len(weak_dimensions) >= 3:
            return "major_restructure"
        elif EvaluationDimension.ACTIONABILITY.value in weak_dimensions:
            return "enhance_actionability"
        elif EvaluationDimension.RISK_ASSESSMENT.value in weak_dimensions:
            return "strengthen_risk_analysis"
        elif EvaluationDimension.ACCURACY.value in weak_dimensions:
            return "improve_accuracy"
        else:
            return "targeted_improvement"
    
    def _apply_optimizations(self, analysis: Dict[str, Any], evaluation: AnalysisEvaluation, strategy: str, symbol: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific optimizations based on strategy."""
        
        optimization_prompts = {
            "major_restructure": f"""
            Completely restructure this {symbol} investment analysis to address major weaknesses:
            
            Original Analysis:
            {json.dumps(analysis, indent=2)[:2500]}...
            
            Key Weaknesses to Address:
            {evaluation.weaknesses}
            
            Improvement Recommendations:
            {evaluation.improvement_recommendations}
            
            Create a substantially improved analysis that addresses all major issues while retaining strengths.
            Provide the complete restructured analysis in the same JSON format.
            """,
            
            "enhance_actionability": f"""
            Enhance the actionability of this {symbol} investment analysis:
            
            Current Analysis:
            {json.dumps(analysis, indent=2)[:2500]}...
            
            Specific Issues with Actionability:
            {[rec for rec in evaluation.improvement_recommendations if 'action' in rec.lower()]}
            
            Make the analysis more actionable by:
            1. Adding specific entry/exit points
            2. Clarifying position sizing recommendations
            3. Providing clear monitoring metrics
            4. Setting specific timelines
            5. Adding contingency plans
            
            Return the enhanced analysis in the same JSON format.
            """,
            
            "strengthen_risk_analysis": f"""
            Strengthen the risk assessment in this {symbol} investment analysis:
            
            Current Analysis:
            {json.dumps(analysis, indent=2)[:2500]}...
            
            Risk Assessment Improvements Needed:
            {[rec for rec in evaluation.improvement_recommendations if 'risk' in rec.lower()]}
            
            Enhance by:
            1. Identifying additional risk factors
            2. Quantifying risks where possible
            3. Adding scenario analyses
            4. Providing risk mitigation strategies
            5. Assessing correlation risks
            
            Return the enhanced analysis in the same JSON format.
            """,
            
            "improve_accuracy": f"""
            Improve the accuracy of this {symbol} investment analysis:
            
            Current Analysis:
            {json.dumps(analysis, indent=2)[:2500]}...
            
            Financial Data for Verification:
            {json.dumps(financial_data, indent=2, default=str)[:1500]}...
            
            Accuracy Issues to Address:
            {[rec for rec in evaluation.improvement_recommendations if 'accur' in rec.lower()]}
            
            Verify and correct:
            1. Financial metrics and ratios
            2. Price targets and valuations
            3. Market characterizations
            4. Data interpretations
            5. Forecast assumptions
            
            Return the corrected analysis in the same JSON format.
            """,
            
            "minor_refinement": f"""
            Make minor refinements to this generally good {symbol} investment analysis:
            
            Current Analysis:
            {json.dumps(analysis, indent=2)[:2500]}...
            
            Minor Improvements Suggested:
            {evaluation.improvement_recommendations[:3]}
            
            Make targeted improvements while preserving the overall quality and structure.
            Return the refined analysis in the same JSON format.
            """
        }
        
        prompt = optimization_prompts.get(strategy, optimization_prompts["minor_refinement"])
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            optimized_text = response.choices[0].message.content
            optimized_analysis = json.loads(optimized_text)
            
            # Add optimization metadata
            optimized_analysis["optimization_metadata"] = {
                "optimized_at": datetime.now().isoformat(),
                "optimization_strategy": strategy,
                "previous_score": evaluation.overall_score,
                "improvements_addressed": evaluation.improvement_recommendations[:5],
                "version": analysis.get("generation_metadata", {}).get("version", "unknown") + "_optimized"
            }
            
            return optimized_analysis
            
        except Exception as e:
            # Return original analysis with error note if optimization fails
            analysis["optimization_error"] = f"Optimization failed: {str(e)}"
            return analysis


class EvaluatorOptimizerWorkflow:
    """
    Complete Evaluator-Optimizer Workflow that iteratively improves analysis quality
    through generate → evaluate → refine cycles.
    """
    
    def __init__(self, model: str = "gpt-4", max_iterations: int = 3, target_score: float = 8.0):
        self.model = model
        self.max_iterations = max_iterations
        self.target_score = target_score
        
        self.generator = AnalysisGenerator(model)
        self.evaluator = AnalysisEvaluator(model)
        self.optimizer = AnalysisOptimizer(model)
        
        self.optimization_history: List[OptimizationIteration] = []
    
    def execute(self, financial_data: Dict[str, Any], symbol: str, research_goal: str) -> Dict[str, Any]:
        """
        Execute the complete evaluator-optimizer workflow.
        
        Args:
            financial_data: Financial data for analysis
            symbol: Stock symbol
            research_goal: Research objective
            
        Returns:
            Optimized analysis with iteration history
        """
        workflow_start = datetime.now()
        
        try:
            # Step 1: Generate initial analysis
            print("Step 1: Generating initial analysis...")
            current_analysis = self.generator.generate_initial_analysis(financial_data, symbol, research_goal)
            
            if "error" in current_analysis:
                return {"error": "Failed to generate initial analysis", "details": current_analysis}
            
            # Iterative optimization loop
            for iteration in range(self.max_iterations):
                print(f"Step {iteration + 2}: Evaluation and optimization iteration {iteration + 1}...")
                
                # Evaluate current analysis
                evaluation = self.evaluator.evaluate_analysis(current_analysis, symbol, financial_data)
                
                # Check if we've reached target quality
                if evaluation.overall_score >= self.target_score and evaluation.meets_threshold:
                    print(f"Target quality reached at iteration {iteration + 1}")
                    break
                
                # Optimize analysis based on evaluation
                if iteration < self.max_iterations - 1:  # Don't optimize on last iteration
                    previous_score = evaluation.overall_score
                    optimized_analysis = self.optimizer.optimize_analysis(
                        current_analysis, evaluation, symbol, financial_data
                    )
                    
                    # Calculate improvement
                    improvement_delta = 0.0  # Will be calculated in next evaluation
                    
                    # Store iteration history
                    iteration_record = OptimizationIteration(
                        iteration=iteration + 1,
                        analysis=current_analysis.copy(),
                        evaluation=evaluation,
                        refinement_strategy=optimized_analysis.get("optimization_metadata", {}).get("optimization_strategy", "unknown"),
                        improvement_delta=improvement_delta
                    )
                    self.optimization_history.append(iteration_record)
                    
                    current_analysis = optimized_analysis
                else:
                    # Final iteration - just store evaluation
                    iteration_record = OptimizationIteration(
                        iteration=iteration + 1,
                        analysis=current_analysis.copy(),
                        evaluation=evaluation,
                        refinement_strategy="final_evaluation",
                        improvement_delta=0.0
                    )
                    self.optimization_history.append(iteration_record)
            
            # Calculate improvement deltas retroactively
            for i in range(len(self.optimization_history) - 1):
                current_score = self.optimization_history[i].evaluation.overall_score
                next_score = self.optimization_history[i + 1].evaluation.overall_score
                self.optimization_history[i].improvement_delta = next_score - current_score
            
            # Final evaluation for the workflow summary
            final_evaluation = self.evaluator.evaluate_analysis(current_analysis, symbol, financial_data)
            
            # Compile workflow results
            workflow_results = {
                "symbol": symbol,
                "workflow_type": "evaluator_optimizer",
                "execution_metadata": {
                    "start_time": workflow_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_processing_time": (datetime.now() - workflow_start).total_seconds(),
                    "iterations_completed": len(self.optimization_history),
                    "target_score_reached": final_evaluation.overall_score >= self.target_score
                },
                "optimization_summary": {
                    "initial_score": self.optimization_history[0].evaluation.overall_score if self.optimization_history else 0.0,
                    "final_score": final_evaluation.overall_score,
                    "total_improvement": final_evaluation.overall_score - (self.optimization_history[0].evaluation.overall_score if self.optimization_history else 0.0),
                    "iterations_used": len(self.optimization_history),
                    "quality_threshold_met": final_evaluation.meets_threshold
                },
                "iteration_history": [
                    {
                        "iteration": it.iteration,
                        "score": it.evaluation.overall_score,
                        "improvement_delta": it.improvement_delta,
                        "strategy": it.refinement_strategy,
                        "meets_threshold": it.evaluation.meets_threshold
                    }
                    for it in self.optimization_history
                ],
                "final_analysis": current_analysis,
                "final_evaluation": {
                    "overall_score": final_evaluation.overall_score,
                    "dimension_scores": {dim.value: score.score for dim, score in final_evaluation.dimension_scores.items()},
                    "strengths": final_evaluation.strengths,
                    "weaknesses": final_evaluation.weaknesses,
                    "meets_threshold": final_evaluation.meets_threshold
                },
                "quality_progression": self._analyze_quality_progression(),
                "optimized_investment_recommendation": self._extract_final_recommendation(current_analysis, final_evaluation)
            }
            
            return workflow_results
            
        except Exception as e:
            return {
                "symbol": symbol,
                "workflow_type": "evaluator_optimizer",
                "error": f"Workflow execution failed: {str(e)}",
                "execution_metadata": {
                    "start_time": workflow_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "failed_at": "unknown"
                }
            }
    
    def _analyze_quality_progression(self) -> Dict[str, Any]:
        """Analyze how quality improved across iterations."""
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        scores = [it.evaluation.overall_score for it in self.optimization_history]
        improvements = [it.improvement_delta for it in self.optimization_history[:-1]]
        
        return {
            "score_progression": scores,
            "improvement_per_iteration": improvements,
            "total_improvement": scores[-1] - scores[0] if len(scores) > 1 else 0.0,
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
            "convergence_pattern": "improving" if all(imp >= 0 for imp in improvements) else "mixed",
            "optimization_efficiency": len([imp for imp in improvements if imp > 0.5]) / len(improvements) if improvements else 0.0
        }
    
    def _extract_final_recommendation(self, analysis: Dict[str, Any], evaluation: AnalysisEvaluation) -> Dict[str, Any]:
        """Extract and enhance the final investment recommendation."""
        
        executive_summary = analysis.get("executive_summary", {})
        investment_strategy = analysis.get("investment_strategy", {})
        
        return {
            "recommendation": executive_summary.get("recommendation", "hold"),
            "confidence": executive_summary.get("confidence_level", 5.0),
            "target_price": executive_summary.get("target_price"),
            "time_horizon": executive_summary.get("time_horizon", "medium_term"),
            "quality_score": evaluation.overall_score,
            "quality_assurance": "high" if evaluation.overall_score >= 8.0 else "medium" if evaluation.overall_score >= 6.5 else "low",
            "key_strengths": evaluation.strengths[:3],
            "remaining_concerns": evaluation.weaknesses[:2] if evaluation.weaknesses else [],
            "actionability_score": evaluation.dimension_scores.get(EvaluationDimension.ACTIONABILITY, EvaluationScore(EvaluationDimension.ACTIONABILITY, 5.0, "", [])).score,
            "investment_thesis": executive_summary.get("investment_thesis", ""),
            "entry_strategy": investment_strategy.get("entry_strategy", "gradual"),
            "monitoring_points": investment_strategy.get("monitoring_points", [])
        }