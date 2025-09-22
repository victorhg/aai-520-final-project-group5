# REFERENCE BRANCH - One SHOT - Use only for reference

# Investment Research Agent - AAI-520 Final Project

An autonomous AI agent that researches financial markets using advanced workflow patterns and demonstrates sophisticated agent capabilities.

## 🎯 Project Overview

This project implements an **Autonomous Investment Research Agent** that demonstrates all required capabilities for the AAI-520 final project:

### 🤖 Agent Functions (33.8%)
✅ **Plans research steps** autonomously for any stock symbol  
✅ **Uses tools dynamically** (APIs, datasets, retrieval systems)  
✅ **Self-reflects** to assess output quality across multiple dimensions  
✅ **Learns across runs** through persistent memory and experience synthesis  

### 🔄 Workflow Patterns (33.8%)
✅ **Prompt Chaining**: News → Preprocess → Classify → Extract → Summarize  
✅ **Routing**: Content directed to specialist agents (earnings, news, market)  
✅ **Evaluator-Optimizer**: Generate → Evaluate → Refine analysis quality  

## 🏗️ Architecture

```
src/
├── agents/
│   ├── base_agent.py              # Core autonomous agent framework
│   └── investment_research_agent.py # Main research agent
├── workflows/
│   ├── prompt_chaining.py         # News processing pipeline
│   ├── routing.py                 # Specialist agent routing
│   └── evaluator_optimizer.py    # Quality improvement loop
├── tools/
│   └── financial_data_tools.py   # Data collection tools
├── config.py                     # Configuration management
└── main.py                       # Entry point and demonstrations
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys
- **OpenAI API Key** (Required): For LLM-powered analysis
- **Alpha Vantage API Key** (Optional): For additional financial data
- **News API Key** (Optional): For enhanced news coverage

### Run Demonstrations
```bash
# Run full capability demonstrations
python src/main.py

# Interactive research mode
python src/main.py --interactive
```

## 🤖 Agent Capabilities Demonstrated

### 1. Autonomous Planning
- Creates multi-step research plans for any stock symbol
- Identifies required tools and resources
- Estimates time and complexity
- Adapts plans based on available data

### 2. Dynamic Tool Usage
- **Stock Data Tool**: Price, volume, financial metrics
- **News Data Tool**: Articles, sentiment, RSS feeds  
- **Earnings Data Tool**: Financial statements, analyst estimates
- **Market Data Tool**: Indices, economic indicators
- **Alpha Vantage Tool**: Fundamental data and technical indicators

### 3. Self-Reflection
- **Quality Assessment**: Accuracy, completeness, actionability
- **Strategy Evaluation**: Tool selection, process effectiveness
- **Learning Synthesis**: Knowledge extraction for future use

### 4. Cross-Session Learning
- Persistent memory system stores insights
- Applies past learnings to new research
- Improves strategy over time
- Tracks performance metrics

## 🔄 Workflow Pattern Details

### Prompt Chaining Workflow
```
News Data → Preprocess → Classify → Extract → Summarize
```
- **Ingest**: Structure raw news data
- **Preprocess**: Clean, tokenize, sentiment analysis
- **Classify**: Categorize by relevance and impact
- **Extract**: Pull structured insights and signals
- **Summarize**: Generate actionable intelligence

### Routing Workflow
```
Financial Data → Content Router → Specialist Agents → Synthesis
```
- **Earnings Analyst**: Financial statements, metrics, valuations
- **News Analyst**: Sentiment, narratives, market impact
- **Market Analyst**: Technical patterns, market context
- **Intelligent Routing**: Directs content to appropriate specialists

### Evaluator-Optimizer Workflow
```
Generate Analysis → Evaluate Quality → Refine → Repeat
```
- **Generation**: Create initial investment analysis
- **Evaluation**: Multi-dimensional quality assessment
- **Optimization**: Targeted improvements based on feedback
- **Iteration**: Repeat until quality threshold met

## 📊 Example Output

### Agent Functions Evidence
```
🤖 AGENT FUNCTIONS DEMONSTRATED:
   ✓ Autonomous Planning: 5 steps planned
   ✓ Dynamic Tool Usage: 4 tools used adaptively
   ✓ Self-Reflection: 3 reflection types completed
   ✓ Cross-Session Learning: 15 memory entries applied
```

### Workflow Patterns Evidence
```
🔄 WORKFLOW PATTERNS DEMONSTRATED:
   ✓ Prompt Chaining: News → Process → Analyze
   ✓ Routing: 3 specialists engaged
   ✓ Evaluator-Optimizer: 3 improvement iterations
```

### Investment Analysis Output
```json
{
  "investment_recommendation": {
    "action": "buy",
    "conviction_level": "high",
    "target_price": 185.50,
    "time_horizon": "medium_term"
  },
  "quality_score": 8.7,
  "confidence": 0.89
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_key

# Optional
ALPHA_VANTAGE_API_KEY=your_av_key
NEWS_API_KEY=your_news_key

# Agent Configuration
DEFAULT_MODEL=gpt-4
TEMPERATURE=0.7
TARGET_QUALITY_SCORE=8.0
MAX_ITERATIONS=3
```

### Memory System
- Persistent JSON storage of experiences
- Quality-based memory filtering
- Cross-session learning capabilities
- Performance tracking over time

## 📈 Performance Metrics

The agent tracks comprehensive performance metrics:
- **Quality Scores**: Multi-dimensional analysis evaluation
- **Processing Time**: Efficiency of research workflows
- **Learning Progression**: Improvement over time
- **Autonomous Behavior**: Evidence of independent operation

## 🎓 Educational Value

This project demonstrates advanced AI concepts:
- **Autonomous Agent Design**: Planning, execution, reflection
- **Workflow Orchestration**: Complex multi-step processes
- **Quality Assurance**: Iterative improvement systems
- **Memory and Learning**: Persistent knowledge building
- **Tool Integration**: Dynamic resource utilization

## 🔍 Testing Different Stocks

The agent can research any publicly traded stock:
```python
# Examples that demonstrate different scenarios
agent.research_stock("AAPL")  # Large cap tech
agent.research_stock("TSLA")  # Volatile growth stock  
agent.research_stock("BRK-A") # Value investing classic
agent.research_stock("NVDA")  # AI/semiconductor play
```

## 🚨 Limitations & Disclaimers

- **Educational Purpose**: This is a learning project, not financial advice
- **Data Dependencies**: Requires API access for real-time data
- **Market Conditions**: Performance may vary with market volatility
- **AI Limitations**: Subject to LLM capabilities and biases

## 📚 Future Enhancements

Potential improvements for production use:
- Real-time data streaming
- Advanced technical analysis
- Portfolio optimization
- Risk management integration
- Multi-asset class support
- Backtesting capabilities

## 🤝 Contributing

This is an academic project for AAI-520. The implementation showcases:
- Clean, modular architecture
- Comprehensive documentation
- Robust error handling
- Extensible design patterns

## 📄 License

Academic use only - AAI-520 Final Project

---

## Group Members

- Antonio Recalde
- Ajmal Jalal
- Darin Verduzzo
- Michael De Leon
- Victor H. Germano



## Files

[Google Drive Folder](https://drive.google.com/drive/folders/1G80xy8F7216N3mFkswAjnpRopSZasiTw)


## Objective



## References



## Technical Details
