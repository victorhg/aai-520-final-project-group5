"""
Routing Workflow: Direct content to specialized analyst agents

This workflow demonstrates intelligent routing of different types of financial content
to appropriate specialist agents (earnings, news, market analysis) for targeted analysis.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import openai


class ContentType(Enum):
    """Types of financial content that can be routed."""
    EARNINGS = "earnings"
    NEWS = "news"
    MARKET_DATA = "market_data"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


@dataclass
class AnalysisRequest:
    """Request for analysis to be routed to specialist."""
    content_type: ContentType
    data: Dict[str, Any]
    symbol: str
    priority: int = 5  # 1-10 scale
    context: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class AnalysisResponse:
    """Response from specialist agent."""
    analyst_type: str
    analysis: Dict[str, Any]
    confidence: float
    processing_time: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class SpecialistAgent(ABC):
    """Abstract base class for specialist analyst agents."""
    
    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.specialization = []
        self.confidence_threshold = 0.7
    
    @property
    @abstractmethod
    def supported_content_types(self) -> List[ContentType]:
        """Content types this specialist can analyze."""
        pass
    
    @abstractmethod
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform specialized analysis on the request."""
        pass
    
    def can_handle(self, content_type: ContentType) -> bool:
        """Check if this specialist can handle the content type."""
        return content_type in self.supported_content_types
    
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Helper method to call the language model."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"


class EarningsAnalyst(SpecialistAgent):
    """Specialist agent for analyzing earnings data and financial statements."""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("EarningsAnalyst", model)
        self.specialization = ["financial_statements", "earnings_calls", "guidance", "metrics"]
    
    @property
    def supported_content_types(self) -> List[ContentType]:
        return [ContentType.EARNINGS, ContentType.FUNDAMENTAL_ANALYSIS]
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze earnings and fundamental data."""
        start_time = datetime.now()
        
        # Extract relevant data
        earnings_data = request.data
        symbol = request.symbol
        
        analysis = self._perform_earnings_analysis(earnings_data, symbol)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            analyst_type="EarningsAnalyst",
            analysis=analysis,
            confidence=analysis.get("confidence", 0.8),
            processing_time=processing_time,
            recommendations=analysis.get("recommendations", []),
            metadata={
                "specialist": self.name,
                "specialization": self.specialization,
                "processed_at": datetime.now().isoformat()
            }
        )
    
    def _perform_earnings_analysis(self, earnings_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Perform detailed earnings analysis."""
        
        prompt = f"""
        As a specialist earnings analyst, provide a comprehensive analysis of {symbol}'s financial data:
        
        Earnings Data:
        {json.dumps(earnings_data, indent=2, default=str)}
        
        Provide analysis in JSON format covering:
        {{
            "financial_health": {{
                "overall_score": 8.5,  # 1-10 scale
                "revenue_trend": "growing|declining|stable",
                "profitability_trend": "improving|declining|stable",
                "debt_levels": "healthy|concerning|critical",
                "cash_position": "strong|adequate|weak"
            }},
            "key_metrics_analysis": {{
                "revenue_growth": {{"value": "15.2%", "assessment": "strong"}},
                "margin_expansion": {{"value": "2.1%", "assessment": "positive"}},
                "eps_growth": {{"value": "23.4%", "assessment": "excellent"}},
                "roe": {{"value": "18.5%", "assessment": "strong"}},
                "debt_to_equity": {{"value": "0.45", "assessment": "healthy"}}
            }},
            "earnings_quality": {{
                "recurring_revenue_pct": 75.5,
                "one_time_items": ["item1", "item2"],
                "accounting_quality": "high|medium|low",
                "guidance_reliability": "high|medium|low"
            }},
            "comparative_analysis": {{
                "vs_sector": "outperforming|inline|underperforming",
                "vs_peers": "better|similar|worse",
                "vs_expectations": "beat|met|missed",
                "historical_performance": "improving|stable|declining"
            }},
            "forward_outlook": {{
                "revenue_forecast": "positive|neutral|negative",
                "margin_outlook": "expanding|stable|contracting",
                "guidance_changes": "raised|maintained|lowered",
                "key_drivers": ["driver1", "driver2", "driver3"]
            }},
            "risk_assessment": {{
                "earnings_volatility": "low|medium|high",
                "sector_headwinds": ["headwind1", "headwind2"],
                "company_specific_risks": ["risk1", "risk2"],
                "regulatory_risks": ["risk1", "risk2"]
            }},
            "valuation_indicators": {{
                "pe_assessment": "undervalued|fairly_valued|overvalued",
                "peg_ratio": "attractive|fair|expensive",
                "price_to_book": "cheap|reasonable|expensive",
                "enterprise_value": "undervalued|fairly_valued|overvalued"
            }},
            "recommendations": [
                "Specific actionable recommendation 1",
                "Specific actionable recommendation 2",
                "Specific actionable recommendation 3"
            ],
            "confidence": 0.85,  # 0-1 scale for analysis confidence
            "key_takeaways": [
                "Most important insight 1",
                "Most important insight 2",
                "Most important insight 3"
            ]
        }}
        
        Focus on providing quantitative analysis where possible and clear investment implications.
        """
        
        response = self._call_llm(prompt, temperature=0.2)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse earnings analysis",
                "raw_response": response,
                "confidence": 0.3
            }


class NewsAnalyst(SpecialistAgent):
    """Specialist agent for analyzing news and sentiment."""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("NewsAnalyst", model)
        self.specialization = ["news_analysis", "sentiment", "market_impact", "narrative"]
    
    @property
    def supported_content_types(self) -> List[ContentType]:
        return [ContentType.NEWS, ContentType.SENTIMENT_ANALYSIS]
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze news content for investment implications."""
        start_time = datetime.now()
        
        news_data = request.data
        symbol = request.symbol
        
        analysis = self._perform_news_analysis(news_data, symbol)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            analyst_type="NewsAnalyst",
            analysis=analysis,
            confidence=analysis.get("confidence", 0.75),
            processing_time=processing_time,
            recommendations=analysis.get("recommendations", []),
            metadata={
                "specialist": self.name,
                "specialization": self.specialization,
                "processed_at": datetime.now().isoformat()
            }
        )
    
    def _perform_news_analysis(self, news_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Perform specialized news and sentiment analysis."""
        
        prompt = f"""
        As a specialist news analyst, analyze the news sentiment and market implications for {symbol}:
        
        News Data:
        {json.dumps(news_data, indent=2, default=str)}
        
        Provide analysis in JSON format:
        {{
            "sentiment_analysis": {{
                "overall_sentiment": "bullish|bearish|neutral",
                "sentiment_strength": 7.5,  # 1-10 scale
                "sentiment_trend": "improving|stable|deteriorating",
                "news_volume": "high|normal|low",
                "source_credibility": "high|medium|low"
            }},
            "narrative_themes": {{
                "dominant_narratives": ["narrative1", "narrative2", "narrative3"],
                "emerging_themes": ["theme1", "theme2"],
                "narrative_shift": "positive|negative|none",
                "media_attention": "increasing|stable|decreasing"
            }},
            "market_impact_assessment": {{
                "immediate_impact": "high|medium|low",
                "short_term_impact": "high|medium|low",
                "long_term_impact": "high|medium|low",
                "volatility_expectation": "high|medium|low",
                "trading_implications": ["implication1", "implication2"]
            }},
            "catalyst_identification": {{
                "positive_catalysts": ["catalyst1", "catalyst2"],
                "negative_catalysts": ["catalyst1", "catalyst2"],
                "upcoming_events": ["event1", "event2"],
                "market_moving_potential": "high|medium|low"
            }},
            "stakeholder_sentiment": {{
                "institutional_investors": "positive|neutral|negative",
                "retail_investors": "positive|neutral|negative",
                "analysts": "positive|neutral|negative",
                "management": "confident|neutral|concerned"
            }},
            "competitive_intelligence": {{
                "competitor_mentions": ["competitor1", "competitor2"],
                "industry_trends": ["trend1", "trend2"],
                "market_positioning": "improving|stable|declining",
                "competitive_advantages": ["advantage1", "advantage2"]
            }},
            "risk_signals": {{
                "reputation_risks": ["risk1", "risk2"],
                "regulatory_concerns": ["concern1", "concern2"],
                "operational_risks": ["risk1", "risk2"],
                "early_warning_signs": ["signal1", "signal2"]
            }},
            "recommendations": [
                "Monitor news sentiment trends closely",
                "Watch for follow-up developments on key themes",
                "Consider sentiment-driven trading opportunities"
            ],
            "confidence": 0.80,
            "key_insights": [
                "Most important news-driven insight 1",
                "Most important news-driven insight 2",
                "Most important news-driven insight 3"
            ]
        }}
        
        Focus on how news sentiment translates to market action and investment opportunities.
        """
        
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse news analysis",
                "raw_response": response,
                "confidence": 0.3
            }


class MarketAnalyst(SpecialistAgent):
    """Specialist agent for market data and technical analysis."""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("MarketAnalyst", model)
        self.specialization = ["market_trends", "technical_indicators", "volume_analysis", "macro_factors"]
    
    @property
    def supported_content_types(self) -> List[ContentType]:
        return [ContentType.MARKET_DATA, ContentType.TECHNICAL_ANALYSIS]
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze market data and technical indicators."""
        start_time = datetime.now()
        
        market_data = request.data
        symbol = request.symbol
        
        analysis = self._perform_market_analysis(market_data, symbol)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            analyst_type="MarketAnalyst",
            analysis=analysis,
            confidence=analysis.get("confidence", 0.75),
            processing_time=processing_time,
            recommendations=analysis.get("recommendations", []),
            metadata={
                "specialist": self.name,
                "specialization": self.specialization,
                "processed_at": datetime.now().isoformat()
            }
        )
    
    def _perform_market_analysis(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Perform specialized market and technical analysis."""
        
        prompt = f"""
        As a specialist market analyst, analyze the market data and technical patterns for {symbol}:
        
        Market Data:
        {json.dumps(market_data, indent=2, default=str)}
        
        Provide analysis in JSON format:
        {{
            "technical_analysis": {{
                "trend": "uptrend|downtrend|sideways",
                "trend_strength": 7.5,  # 1-10 scale
                "support_levels": [120.50, 115.25, 110.00],
                "resistance_levels": [135.75, 140.25, 145.50],
                "momentum": "bullish|bearish|neutral",
                "volume_confirmation": "confirmed|unconfirmed|weak"
            }},
            "price_action": {{
                "current_position": "near_high|mid_range|near_low",
                "volatility_assessment": "high|normal|low",
                "price_patterns": ["pattern1", "pattern2"],
                "breakout_potential": "high|medium|low",
                "key_levels": ["level1", "level2", "level3"]
            }},
            "market_context": {{
                "relative_performance": "outperforming|inline|underperforming",
                "sector_rotation": "in_favor|neutral|out_of_favor",
                "market_sentiment": "risk_on|neutral|risk_off",
                "institutional_flow": "buying|neutral|selling",
                "correlation_analysis": "low|medium|high"
            }},
            "volume_analysis": {{
                "volume_trend": "increasing|stable|decreasing",
                "volume_quality": "healthy|average|poor",
                "distribution_pattern": "accumulation|neutral|distribution",
                "unusual_activity": "yes|no",
                "volume_price_divergence": "positive|negative|none"
            }},
            "macro_factors": {{
                "interest_rate_sensitivity": "high|medium|low",
                "economic_cycle_position": "early|mid|late",
                "sector_tailwinds": ["tailwind1", "tailwind2"],
                "macro_headwinds": ["headwind1", "headwind2"],
                "currency_impact": "positive|negative|neutral"
            }},
            "trading_signals": {{
                "entry_signals": ["signal1", "signal2"],
                "exit_signals": ["signal1", "signal2"],
                "stop_loss_levels": [105.25, 100.50],
                "profit_targets": [145.00, 155.25],
                "position_sizing": "aggressive|moderate|conservative"
            }},
            "market_timing": {{
                "current_phase": "accumulation|markup|distribution|markdown",
                "cycle_position": "early|middle|late",
                "seasonal_factors": ["factor1", "factor2"],
                "event_risk": "high|medium|low",
                "liquidity_conditions": "abundant|normal|tight"
            }},
            "recommendations": [
                "Consider technical breakout above resistance",
                "Monitor volume for confirmation signals",
                "Watch broader market correlation patterns"
            ],
            "confidence": 0.75,
            "key_insights": [
                "Primary technical insight 1",
                "Primary technical insight 2", 
                "Primary technical insight 3"
            ]
        }}
        
        Focus on actionable technical insights and market timing considerations.
        """
        
        response = self._call_llm(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse market analysis",
                "raw_response": response,
                "confidence": 0.3
            }


class ContentRouter:
    """Intelligent router that directs content to appropriate specialist agents."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.specialists: Dict[str, SpecialistAgent] = {}
        self.routing_history: List[Dict[str, Any]] = []
    
    def register_specialist(self, specialist: SpecialistAgent) -> None:
        """Register a specialist agent with the router."""
        self.specialists[specialist.name] = specialist
    
    def route_content(self, data: Dict[str, Any], symbol: str, context: str = "") -> List[AnalysisRequest]:
        """
        Intelligently route content to appropriate specialists.
        
        Args:
            data: Raw data to be analyzed
            symbol: Stock symbol for context
            context: Additional context for routing decisions
            
        Returns:
            List of analysis requests for different specialists
        """
        # Determine content types present in the data
        content_types = self._identify_content_types(data, symbol, context)
        
        # Create analysis requests for each identified content type
        requests = []
        for content_type, content_data in content_types.items():
            request = AnalysisRequest(
                content_type=content_type,
                data=content_data,
                symbol=symbol,
                priority=self._calculate_priority(content_type, content_data),
                context=context,
                metadata={"routing_timestamp": datetime.now().isoformat()}
            )
            requests.append(request)
        
        # Log routing decision
        self.routing_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "identified_content_types": list(content_types.keys()),
            "total_requests": len(requests)
        })
        
        return requests
    
    def execute_analysis(self, requests: List[AnalysisRequest]) -> Dict[str, AnalysisResponse]:
        """
        Execute analysis requests by routing to appropriate specialists.
        
        Args:
            requests: List of analysis requests
            
        Returns:
            Dictionary mapping specialist names to their analysis responses
        """
        responses = {}
        
        for request in requests:
            # Find capable specialists
            capable_specialists = [
                specialist for specialist in self.specialists.values()
                if specialist.can_handle(request.content_type)
            ]
            
            if not capable_specialists:
                # Create a fallback response
                responses[f"NoSpecialist_{request.content_type.value}"] = AnalysisResponse(
                    analyst_type="Unknown",
                    analysis={"error": f"No specialist available for {request.content_type.value}"},
                    confidence=0.0,
                    processing_time=0.0,
                    recommendations=[],
                    metadata={"error": "routing_failed"}
                )
                continue
            
            # Use the first capable specialist (could implement more sophisticated selection)
            specialist = capable_specialists[0]
            
            try:
                response = specialist.analyze(request)
                responses[specialist.name] = response
                
            except Exception as e:
                # Create error response
                responses[specialist.name] = AnalysisResponse(
                    analyst_type=specialist.name,
                    analysis={"error": f"Analysis failed: {str(e)}"},
                    confidence=0.0,
                    processing_time=0.0,
                    recommendations=[],
                    metadata={"error": "analysis_failed"}
                )
        
        return responses
    
    def _identify_content_types(self, data: Dict[str, Any], symbol: str, context: str) -> Dict[ContentType, Dict[str, Any]]:
        """Use LLM to identify what types of content are present in the data."""
        
        prompt = f"""
        Analyze this financial data for {symbol} and identify what types of content are present:
        
        Data structure:
        {json.dumps({k: type(v).__name__ for k, v in data.items()}, indent=2)}
        
        Context: {context}
        
        Available content types:
        - earnings: Financial statements, earnings data, quarterly results
        - news: News articles, press releases, announcements  
        - market_data: Stock prices, volume, indices, market indicators
        - technical_analysis: Price charts, technical indicators, patterns
        - fundamental_analysis: Valuation metrics, financial ratios
        - sentiment_analysis: Social media sentiment, news sentiment
        
        Respond with JSON mapping content types to relevant data portions:
        {{
            "earnings": {{"relevant": true, "data_keys": ["earnings", "financials"]}},
            "news": {{"relevant": false, "data_keys": []}},
            "market_data": {{"relevant": true, "data_keys": ["current_price", "volume"]}}
        }}
        """
        
        try:
            response = self._call_llm(prompt)
            content_mapping = json.loads(response)
            
            # Extract relevant data for each content type
            content_types = {}
            for content_type_str, info in content_mapping.items():
                if info.get("relevant", False):
                    try:
                        content_type = ContentType(content_type_str)
                        relevant_data = {}
                        
                        # Extract relevant data based on identified keys
                        for key in info.get("data_keys", []):
                            if key in data:
                                relevant_data[key] = data[key]
                        
                        # If no specific keys, include all data
                        if not relevant_data:
                            relevant_data = data
                        
                        content_types[content_type] = relevant_data
                        
                    except ValueError:
                        # Skip invalid content types
                        continue
            
            return content_types
            
        except Exception as e:
            # Fallback: try to infer content types based on data keys
            content_types = {}
            
            data_keys = set(data.keys())
            
            # Simple heuristic-based content type detection
            if any(key in data_keys for key in ['earnings', 'financials', 'income_statement', 'balance_sheet']):
                content_types[ContentType.EARNINGS] = data
            
            if any(key in data_keys for key in ['articles', 'news', 'sentiment', 'headlines']):
                content_types[ContentType.NEWS] = data
            
            if any(key in data_keys for key in ['current_price', 'volume', 'indices', 'market_cap']):
                content_types[ContentType.MARKET_DATA] = data
            
            return content_types
    
    def _calculate_priority(self, content_type: ContentType, data: Dict[str, Any]) -> int:
        """Calculate priority for analysis request (1-10 scale)."""
        # Simple priority calculation - can be made more sophisticated
        priority_map = {
            ContentType.EARNINGS: 9,
            ContentType.NEWS: 7,
            ContentType.MARKET_DATA: 6,
            ContentType.TECHNICAL_ANALYSIS: 5,
            ContentType.FUNDAMENTAL_ANALYSIS: 8,
            ContentType.SENTIMENT_ANALYSIS: 4
        }
        
        return priority_map.get(content_type, 5)
    
    def _call_llm(self, prompt: str) -> str:
        """Helper method to call the language model."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"


class RoutingWorkflow:
    """
    Complete Routing Workflow that intelligently directs different types of 
    financial content to appropriate specialist agents for analysis.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.router = ContentRouter(model)
        
        # Register specialist agents
        self.router.register_specialist(EarningsAnalyst(model))
        self.router.register_specialist(NewsAnalyst(model))
        self.router.register_specialist(MarketAnalyst(model))
    
    def execute(self, financial_data: Dict[str, Any], symbol: str, context: str = "") -> Dict[str, Any]:
        """
        Execute the complete routing workflow.
        
        Args:
            financial_data: Complete financial data from various sources
            symbol: Stock symbol for context
            context: Additional context for analysis
            
        Returns:
            Comprehensive analysis from all relevant specialists
        """
        workflow_start = datetime.now()
        
        try:
            # Step 1: Route content to appropriate specialists
            print("Step 1: Routing content to specialists...")
            requests = self.router.route_content(financial_data, symbol, context)
            
            # Step 2: Execute analysis with specialists
            print("Step 2: Executing specialist analyses...")
            specialist_responses = self.router.execute_analysis(requests)
            
            # Step 3: Synthesize specialist insights
            print("Step 3: Synthesizing specialist insights...")
            synthesis = self._synthesize_specialist_insights(specialist_responses, symbol)
            
            # Compile workflow results
            workflow_results = {
                "symbol": symbol,
                "workflow_type": "routing",
                "execution_metadata": {
                    "start_time": workflow_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_processing_time": (datetime.now() - workflow_start).total_seconds(),
                    "specialists_engaged": list(specialist_responses.keys()),
                    "routing_decisions": len(requests)
                },
                "routing_summary": {
                    "content_types_identified": [req.content_type.value for req in requests],
                    "specialists_used": list(specialist_responses.keys()),
                    "total_analyses": len(specialist_responses)
                },
                "specialist_analyses": specialist_responses,
                "synthesized_insights": synthesis,
                "investment_recommendation": synthesis.get("final_recommendation", {})
            }
            
            return workflow_results
            
        except Exception as e:
            return {
                "symbol": symbol,
                "workflow_type": "routing",
                "error": f"Routing workflow failed: {str(e)}",
                "execution_metadata": {
                    "start_time": workflow_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "failed_at": "unknown"
                }
            }
    
    def _synthesize_specialist_insights(self, specialist_responses: Dict[str, AnalysisResponse], symbol: str) -> Dict[str, Any]:
        """Synthesize insights from all specialist analyses into unified recommendation."""
        
        # Extract key insights from each specialist
        specialist_summaries = {}
        for specialist_name, response in specialist_responses.items():
            specialist_summaries[specialist_name] = {
                "key_insights": response.analysis.get("key_insights", []),
                "recommendations": response.recommendations,
                "confidence": response.confidence,
                "specialist_type": response.analyst_type
            }
        
        prompt = f"""
        Synthesize these specialist analyses for {symbol} into a unified investment recommendation:
        
        Specialist Analyses:
        {json.dumps(specialist_summaries, indent=2)}
        
        Create a comprehensive synthesis in JSON format:
        {{
            "cross_specialist_insights": {{
                "consensus_themes": ["theme1", "theme2", "theme3"],
                "conflicting_views": ["conflict1", "conflict2"],
                "confidence_weighted_score": 7.8,
                "specialist_agreement_level": "high|medium|low"
            }},
            "integrated_analysis": {{
                "fundamental_outlook": "positive|neutral|negative",
                "technical_outlook": "bullish|neutral|bearish", 
                "sentiment_outlook": "optimistic|neutral|pessimistic",
                "news_flow_impact": "positive|neutral|negative",
                "overall_direction": "buy|hold|sell"
            }},
            "risk_reward_assessment": {{
                "upside_potential": "high|medium|low",
                "downside_risk": "high|medium|low",
                "risk_adjusted_return": "attractive|fair|poor",
                "time_horizon": "short_term|medium_term|long_term",
                "volatility_expectation": "high|medium|low"
            }},
            "final_recommendation": {{
                "action": "strong_buy|buy|hold|sell|strong_sell",
                "conviction_level": "high|medium|low",
                "position_size": "overweight|market_weight|underweight",
                "target_price": 145.50,
                "stop_loss": 105.25,
                "investment_horizon": "3-6 months"
            }},
            "key_monitoring_points": [
                "Monitor earnings guidance updates",
                "Watch for technical breakout confirmation",
                "Track news sentiment momentum"
            ],
            "specialist_contribution_summary": {{
                "earnings_analyst": "Financial health assessment and valuation insights",
                "news_analyst": "Sentiment trends and narrative analysis", 
                "market_analyst": "Technical patterns and market timing"
            }},
            "synthesis_confidence": 0.82,
            "next_review_date": "2024-01-15"
        }}
        
        Focus on creating actionable, well-reasoned investment guidance that leverages all specialist perspectives.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            synthesis_text = response.choices[0].message.content
            synthesis = json.loads(synthesis_text)
            
            # Add metadata
            synthesis["synthesis_metadata"] = {
                "synthesized_at": datetime.now().isoformat(),
                "specialists_included": list(specialist_responses.keys()),
                "total_specialist_analyses": len(specialist_responses),
                "synthesis_version": "routing_v1.0"
            }
            
            return synthesis
            
        except Exception as e:
            return {
                "error": f"Synthesis failed: {str(e)}",
                "fallback_summary": {
                    "specialists_consulted": list(specialist_responses.keys()),
                    "recommendation": "Further analysis required"
                },
                "synthesis_metadata": {
                    "synthesized_at": datetime.now().isoformat(),
                    "synthesis_version": "routing_v1.0_fallback"
                }
            }