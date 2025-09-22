"""
Prompt Chaining Workflow: News → Preprocess → Classify → Extract → Summarize

This workflow demonstrates sequential processing of financial news through
multiple specialized prompts to extract actionable investment insights.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import openai
from textblob import TextBlob
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


@dataclass
class NewsArticle:
    """Represents a news article with metadata."""
    title: str
    content: str
    source: str
    published_date: str
    url: str = ""
    

@dataclass
class ProcessedNews:
    """Represents news at different stages of processing."""
    original: NewsArticle
    preprocessed: Dict[str, Any]
    classified: Dict[str, Any]
    extracted: Dict[str, Any]
    summarized: Dict[str, Any]


class NewsIngestStep:
    """Step 1: Ingest news from various sources and structure the data."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def execute(self, news_data: Dict[str, Any], symbol: str) -> List[NewsArticle]:
        """
        Ingest and structure news articles from raw data.
        
        Args:
            news_data: Raw news data from news tools
            symbol: Stock symbol for context
        """
        articles = []
        
        for article_data in news_data.get("articles", []):
            # Extract and clean article content
            title = self._clean_text(article_data.get("title", ""))
            summary = self._clean_text(article_data.get("summary", ""))
            
            # Use LLM to extract full content if only summary is available
            if len(summary) < 200 and title:
                content = self._expand_content(title, summary, symbol)
            else:
                content = summary
            
            article = NewsArticle(
                title=title,
                content=content,
                source=article_data.get("source", "Unknown"),
                published_date=article_data.get("published", ""),
                url=article_data.get("link", "")
            )
            
            articles.append(article)
        
        return articles
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,;:!?\'"()-]', '', text)
        
        return text
    
    def _expand_content(self, title: str, summary: str, symbol: str) -> str:
        """Use LLM to expand brief content into more detailed analysis."""
        prompt = f"""
        Based on this news headline and brief summary, provide a more detailed analysis 
        of what this means for {symbol} stock:
        
        Headline: {title}
        Summary: {summary}
        
        Provide a detailed explanation (200-400 words) covering:
        1. What happened and why it's significant
        2. Potential impact on the company's business
        3. Implications for stock performance
        4. Key factors investors should consider
        
        Write in a professional, analytical tone suitable for investment research.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except:
            return summary


class NewsPreprocessStep:
    """Step 2: Preprocess news articles for analysis."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def execute(self, articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """
        Preprocess news articles for further analysis.
        
        Args:
            articles: List of NewsArticle objects
        """
        preprocessed_articles = []
        
        for article in articles:
            processed = self._preprocess_article(article)
            preprocessed_articles.append(processed)
        
        return preprocessed_articles
    
    def _preprocess_article(self, article: NewsArticle) -> Dict[str, Any]:
        """Preprocess a single article."""
        # Tokenize text
        sentences = sent_tokenize(article.content)
        words = word_tokenize(article.content.lower())
        
        # Remove stopwords and get key terms
        filtered_words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Extract key phrases (simple approach)
        key_phrases = self._extract_key_phrases(article.content)
        
        # Perform sentiment analysis
        blob = TextBlob(article.content)
        sentiment_polarity = blob.sentiment.polarity  # -1 to 1
        sentiment_subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Determine sentiment label
        if sentiment_polarity > 0.1:
            sentiment_label = "positive"
        elif sentiment_polarity < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        # Calculate readability metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            "article": article,
            "sentences": sentences,
            "word_tokens": words,
            "filtered_words": filtered_words,
            "key_phrases": key_phrases,
            "sentiment": {
                "polarity": sentiment_polarity,
                "subjectivity": sentiment_subjectivity,
                "label": sentiment_label
            },
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text using simple NLP techniques."""
        # Simple approach: find noun phrases and important terms
        blob = TextBlob(text)
        noun_phrases = list(blob.noun_phrases)
        
        # Filter and rank noun phrases
        key_phrases = []
        for phrase in noun_phrases:
            if len(phrase.split()) >= 2 and len(phrase) > 3:
                key_phrases.append(phrase)
        
        # Return top phrases
        return key_phrases[:max_phrases]


class NewsClassifyStep:
    """Step 3: Classify news articles by relevance and category."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def execute(self, preprocessed_articles: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
        """
        Classify preprocessed articles.
        
        Args:
            preprocessed_articles: List of preprocessed article data
            symbol: Stock symbol for context
        """
        classified_articles = []
        
        for article_data in preprocessed_articles:
            classification = self._classify_article(article_data, symbol)
            article_data["classification"] = classification
            classified_articles.append(article_data)
        
        return classified_articles
    
    def _classify_article(self, article_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Classify a single article using LLM."""
        article = article_data["article"]
        sentiment = article_data["sentiment"]
        
        prompt = f"""
        Classify this financial news article about {symbol}:
        
        Title: {article.title}
        Content: {article.content[:1000]}...
        Detected Sentiment: {sentiment['label']} (polarity: {sentiment['polarity']:.2f})
        
        Provide classification in JSON format:
        {{
            "relevance_score": 8.5,  # 1-10 scale for relevance to {symbol}
            "category": "earnings|product|management|market|regulatory|other",
            "impact_level": "high|medium|low",
            "time_sensitivity": "immediate|short_term|long_term",
            "market_moving": true/false,
            "key_topics": ["topic1", "topic2", "topic3"],
            "confidence": 0.85,  # Confidence in classification
            "reasoning": "Brief explanation of classification decisions"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            classification_text = response.choices[0].message.content
            classification = json.loads(classification_text)
            
            return classification
            
        except Exception as e:
            # Fallback classification
            return {
                "relevance_score": 5.0,
                "category": "other",
                "impact_level": "medium",
                "time_sensitivity": "short_term",
                "market_moving": False,
                "key_topics": article_data.get("key_phrases", [])[:3],
                "confidence": 0.3,
                "reasoning": f"Classification failed: {str(e)}"
            }


class NewsExtractStep:
    """Step 4: Extract structured information and insights."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def execute(self, classified_articles: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
        """
        Extract structured insights from classified articles.
        
        Args:
            classified_articles: List of classified article data
            symbol: Stock symbol for context
        """
        extracted_articles = []
        
        for article_data in classified_articles:
            extraction = self._extract_insights(article_data, symbol)
            article_data["extraction"] = extraction
            extracted_articles.append(article_data)
        
        return extracted_articles
    
    def _extract_insights(self, article_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Extract structured insights from a single article."""
        article = article_data["article"]
        classification = article_data["classification"]
        
        prompt = f"""
        Extract structured investment insights from this {symbol} news article:
        
        Title: {article.title}
        Content: {article.content}
        Category: {classification['category']}
        Impact Level: {classification['impact_level']}
        
        Extract the following information in JSON format:
        {{
            "key_facts": ["fact1", "fact2", "fact3"],
            "financial_metrics": {{
                "revenue_impact": "positive|negative|neutral|unknown",
                "profit_impact": "positive|negative|neutral|unknown",
                "mentioned_numbers": ["number with context"],
                "financial_guidance": "guidance information if any"
            }},
            "business_impact": {{
                "operational_changes": ["change1", "change2"],
                "strategic_implications": ["implication1", "implication2"],
                "competitive_position": "stronger|weaker|unchanged|unknown",
                "market_opportunities": ["opportunity1", "opportunity2"]
            }},
            "investment_signals": {{
                "bullish_indicators": ["indicator1", "indicator2"],
                "bearish_indicators": ["indicator1", "indicator2"],
                "risk_factors": ["risk1", "risk2"],
                "catalysts": ["catalyst1", "catalyst2"]
            }},
            "stakeholder_impact": {{
                "shareholders": "positive|negative|neutral",
                "customers": "positive|negative|neutral",
                "employees": "positive|negative|neutral",
                "regulators": "positive|negative|neutral"
            }},
            "actionable_insights": ["insight1", "insight2", "insight3"]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            extraction_text = response.choices[0].message.content
            extraction = json.loads(extraction_text)
            
            return extraction
            
        except Exception as e:
            # Fallback extraction
            return {
                "key_facts": [article.title],
                "financial_metrics": {
                    "revenue_impact": "unknown",
                    "profit_impact": "unknown",
                    "mentioned_numbers": [],
                    "financial_guidance": "None mentioned"
                },
                "business_impact": {
                    "operational_changes": [],
                    "strategic_implications": [],
                    "competitive_position": "unknown",
                    "market_opportunities": []
                },
                "investment_signals": {
                    "bullish_indicators": [],
                    "bearish_indicators": [],
                    "risk_factors": [],
                    "catalysts": []
                },
                "stakeholder_impact": {
                    "shareholders": "neutral",
                    "customers": "neutral",
                    "employees": "neutral",
                    "regulators": "neutral"
                },
                "actionable_insights": [f"Extraction failed: {str(e)}"]
            }


class NewsSummarizeStep:
    """Step 5: Summarize all processed news into actionable investment intelligence."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def execute(self, extracted_articles: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Synthesize all extracted insights into a comprehensive summary.
        
        Args:
            extracted_articles: List of fully processed article data
            symbol: Stock symbol for context
        """
        summary = self._create_comprehensive_summary(extracted_articles, symbol)
        return summary
    
    def _create_comprehensive_summary(self, extracted_articles: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """Create a comprehensive summary of all news insights."""
        
        # Aggregate insights across all articles
        aggregated_insights = self._aggregate_insights(extracted_articles)
        
        prompt = f"""
        Create a comprehensive investment intelligence summary for {symbol} based on recent news analysis:
        
        Aggregated Insights:
        {json.dumps(aggregated_insights, indent=2)}
        
        Number of articles analyzed: {len(extracted_articles)}
        
        Provide a structured summary in JSON format:
        {{
            "executive_summary": "2-3 sentence overview of key developments",
            "sentiment_analysis": {{
                "overall_sentiment": "bullish|bearish|neutral",
                "sentiment_strength": 7.5,  # 1-10 scale
                "sentiment_drivers": ["driver1", "driver2", "driver3"]
            }},
            "key_developments": [
                {{
                    "development": "description",
                    "impact": "high|medium|low",
                    "timeline": "immediate|short_term|long_term"
                }}
            ],
            "investment_thesis": {{
                "bull_case": ["argument1", "argument2", "argument3"],
                "bear_case": ["argument1", "argument2", "argument3"],
                "key_risks": ["risk1", "risk2", "risk3"],
                "catalysts": ["catalyst1", "catalyst2", "catalyst3"]
            }},
            "financial_outlook": {{
                "revenue_outlook": "positive|negative|neutral",
                "profitability_outlook": "positive|negative|neutral",
                "guidance_changes": "raised|lowered|maintained|none",
                "key_metrics_impact": ["metric impact descriptions"]
            }},
            "strategic_implications": {{
                "competitive_position": "strengthening|weakening|stable",
                "market_opportunities": ["opportunity1", "opportunity2"],
                "operational_efficiency": "improving|declining|stable",
                "innovation_pipeline": ["innovation1", "innovation2"]
            }},
            "recommendation": {{
                "action": "buy|hold|sell|watch",
                "confidence": 0.75,  # 0-1 scale
                "price_targets": {{
                    "bull_case": 150.00,
                    "base_case": 125.00,
                    "bear_case": 100.00
                }},
                "investment_horizon": "short_term|medium_term|long_term",
                "key_monitoring_points": ["point1", "point2", "point3"]
            }},
            "news_quality_assessment": {{
                "total_articles": {len(extracted_articles)},
                "high_relevance_articles": "count",
                "source_diversity": "high|medium|low",
                "information_freshness": "recent|moderate|stale",
                "reliability_score": 0.85  # 0-1 scale
            }}
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000
            )
            
            summary_text = response.choices[0].message.content
            summary = json.loads(summary_text)
            
            # Add metadata
            summary["processing_metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "symbol": symbol,
                "total_articles_processed": len(extracted_articles),
                "workflow_version": "prompt_chaining_v1.0"
            }
            
            return summary
            
        except Exception as e:
            # Fallback summary
            return {
                "executive_summary": f"Analysis of {len(extracted_articles)} news articles for {symbol}",
                "error": f"Summary generation failed: {str(e)}",
                "processing_metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "symbol": symbol,
                    "total_articles_processed": len(extracted_articles),
                    "workflow_version": "prompt_chaining_v1.0"
                }
            }
    
    def _aggregate_insights(self, extracted_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate insights across all articles."""
        all_facts = []
        all_bullish = []
        all_bearish = []
        all_risks = []
        all_catalysts = []
        sentiments = []
        relevance_scores = []
        
        for article_data in extracted_articles:
            extraction = article_data.get("extraction", {})
            classification = article_data.get("classification", {})
            sentiment = article_data.get("sentiment", {})
            
            # Collect facts and signals
            all_facts.extend(extraction.get("key_facts", []))
            
            signals = extraction.get("investment_signals", {})
            all_bullish.extend(signals.get("bullish_indicators", []))
            all_bearish.extend(signals.get("bearish_indicators", []))
            all_risks.extend(signals.get("risk_factors", []))
            all_catalysts.extend(signals.get("catalysts", []))
            
            # Collect sentiment and relevance
            sentiments.append(sentiment.get("polarity", 0))
            relevance_scores.append(classification.get("relevance_score", 5.0))
        
        return {
            "aggregated_facts": all_facts,
            "bullish_signals": all_bullish,
            "bearish_signals": all_bearish,
            "risk_factors": all_risks,
            "catalysts": all_catalysts,
            "average_sentiment": sum(sentiments) / len(sentiments) if sentiments else 0,
            "average_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 5.0
        }


class PromptChainingWorkflow:
    """
    Complete Prompt Chaining Workflow that processes news through all steps:
    Ingest → Preprocess → Classify → Extract → Summarize
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.ingest_step = NewsIngestStep(model)
        self.preprocess_step = NewsPreprocessStep()
        self.classify_step = NewsClassifyStep(model)
        self.extract_step = NewsExtractStep(model)
        self.summarize_step = NewsSummarizeStep(model)
    
    def execute(self, news_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Execute the complete prompt chaining workflow.
        
        Args:
            news_data: Raw news data from news collection tools
            symbol: Stock symbol for context
            
        Returns:
            Complete analysis with all intermediate steps and final summary
        """
        workflow_start = datetime.now()
        
        try:
            # Step 1: Ingest
            print("Step 1: Ingesting news articles...")
            articles = self.ingest_step.execute(news_data, symbol)
            
            # Step 2: Preprocess
            print("Step 2: Preprocessing articles...")
            preprocessed = self.preprocess_step.execute(articles)
            
            # Step 3: Classify
            print("Step 3: Classifying articles...")
            classified = self.classify_step.execute(preprocessed, symbol)
            
            # Step 4: Extract
            print("Step 4: Extracting insights...")
            extracted = self.extract_step.execute(classified, symbol)
            
            # Step 5: Summarize
            print("Step 5: Creating summary...")
            summary = self.summarize_step.execute(extracted, symbol)
            
            # Compile complete workflow results
            workflow_results = {
                "symbol": symbol,
                "workflow_type": "prompt_chaining",
                "execution_metadata": {
                    "start_time": workflow_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_processing_time": (datetime.now() - workflow_start).total_seconds(),
                    "articles_processed": len(articles)
                },
                "step_results": {
                    "ingested_articles": len(articles),
                    "preprocessed_articles": len(preprocessed),
                    "classified_articles": len(classified),
                    "extracted_articles": len(extracted),
                    "final_summary": summary
                },
                "detailed_processing": extracted,  # Full details of all processing steps
                "investment_intelligence": summary  # Final actionable intelligence
            }
            
            return workflow_results
            
        except Exception as e:
            return {
                "symbol": symbol,
                "workflow_type": "prompt_chaining",
                "error": f"Workflow execution failed: {str(e)}",
                "execution_metadata": {
                    "start_time": workflow_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "failed_at": "unknown"
                }
            }