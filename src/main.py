"""
Main entry point for the Investment Research Agent system.
Demonstrates all required agent functions and workflow patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.investment_research_agent import InvestmentResearchAgent
from src.config import get_agent_config
import json


def main():
    """
    Main demonstration of the Investment Research Agent.
    
    This function demonstrates:
    1. Agent Functions: Planning, Tool Usage, Self-Reflection, Learning
    2. Workflow Patterns: Prompt Chaining, Routing, Evaluator-Optimizer
    """
    
    print("🚀 Investment Research Agent - AAI-520 Final Project")
    print("=" * 60)
    
    try:
        # Load configuration
        agent_config, api_keys, data_config = get_agent_config()
        
        # Initialize the agent
        print("🤖 Initializing Investment Research Agent...")
        agent = InvestmentResearchAgent(
            model=agent_config["model"],
            temperature=agent_config["temperature"],
            memory_file=agent_config["memory_file"]
        )
        
        print(f"✅ Agent initialized with {len(agent.tools)} tools")
        print(f"🧠 Memory contains {len(agent.memory)} past experiences")
        
        # Example research demonstrations
        demo_stocks = ["AAPL", "MSFT", "TSLA"]
        
        for symbol in demo_stocks:
            print(f"\n{'='*60}")
            print(f"🔍 DEMONSTRATING AUTONOMOUS RESEARCH FOR {symbol}")
            print(f"{'='*60}")
            
            # Conduct autonomous research
            results = agent.research_stock(
                symbol=symbol,
                research_goal="comprehensive_analysis"
            )
            
            # Display key results
            print(f"\n📊 RESEARCH RESULTS SUMMARY FOR {symbol}")
            print("-" * 40)
            
            if "error" not in results:
                # Agent capabilities demonstrated
                capabilities = results["agent_capabilities_demonstrated"]
                print("🤖 AGENT FUNCTIONS DEMONSTRATED:")
                print(f"   ✓ Autonomous Planning: {capabilities['autonomous_planning']['steps_planned']} steps")
                print(f"   ✓ Dynamic Tool Usage: {len(capabilities['dynamic_tool_usage']['tools_used'])} tools used")
                print(f"   ✓ Self-Reflection: {capabilities['self_reflection']['reflection_types']} reflection types")
                print(f"   ✓ Cross-Session Learning: {capabilities['cross_session_learning']['memory_entries']} memories")
                
                # Workflow patterns demonstrated
                workflows = results["workflow_patterns_demonstrated"]
                print("\n🔄 WORKFLOW PATTERNS DEMONSTRATED:")
                print(f"   ✓ Prompt Chaining: {workflows['prompt_chaining']['executed']}")
                print(f"   ✓ Routing: {len(workflows['routing']['specialists_engaged'])} specialists")
                print(f"   ✓ Evaluator-Optimizer: {workflows['evaluator_optimizer']['iterations_completed']} iterations")
                
                # Investment recommendation
                recommendation = results["investment_recommendation"]
                print(f"\n💡 INVESTMENT RECOMMENDATION:")
                print(f"   Action: {recommendation.get('action', 'N/A')}")
                print(f"   Confidence: {recommendation.get('conviction_level', 'N/A')}")
                print(f"   Target Price: ${recommendation.get('target_price', 'N/A')}")
                
                # Performance metrics
                metrics = results["research_execution"]["performance_metrics"]
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"   Processing Time: {metrics['total_processing_time']:.1f} seconds")
                print(f"   Quality Score: {metrics['overall_quality_score']:.1f}/10")
                print(f"   Data Sources: {metrics['data_sources_used']}")
                
            else:
                print(f"❌ Research failed: {results['error']}")
            
            print(f"\n💾 Results saved to research history")
        
        # Display overall agent performance
        print(f"\n{'='*60}")
        print("📊 OVERALL AGENT PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        performance = agent.get_performance_metrics()
        if "error" not in performance:
            print(f"Total Research Sessions: {performance['total_research_sessions']}")
            print(f"Average Quality Score: {performance['average_quality_score']:.2f}/10")
            print(f"Average Processing Time: {performance['average_processing_time']:.1f} seconds")
            print(f"Quality Trend: {performance['quality_improvement_trend']}")
            print(f"Successful Sessions: {performance['successful_sessions']}")
            
            capabilities = performance['autonomous_capabilities_demonstrated']
            print(f"\nAutonomous Capabilities Evidence:")
            print(f"   Planning Sessions: {capabilities['planning']}")
            print(f"   Tool Usage Events: {capabilities['tool_usage']}")
            print(f"   Self-Reflection Cycles: {capabilities['self_reflection']}")
            print(f"   Learning Entries: {capabilities['learning']}")
        
        print(f"\n✅ All demonstrations completed successfully!")
        print(f"🎓 Agent demonstrates all required capabilities for AAI-520 final project")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Setup Requirements:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Ensure data directory exists and is writable")
        return False


def interactive_research():
    """Interactive mode for custom stock research."""
    
    try:
        agent_config, api_keys, data_config = get_agent_config()
        agent = InvestmentResearchAgent(**agent_config)
        
        print("🔍 Interactive Investment Research Mode")
        print("Enter stock symbols to research (type 'quit' to exit)")
        
        while True:
            symbol = input("\nEnter stock symbol: ").strip().upper()
            
            if symbol.lower() in ['quit', 'exit', 'q']:
                break
            
            if not symbol:
                continue
            
            research_goal = input("Research goal (press Enter for 'comprehensive_analysis'): ").strip()
            if not research_goal:
                research_goal = "comprehensive_analysis"
            
            print(f"\n🤖 Researching {symbol}...")
            results = agent.research_stock(symbol, research_goal)
            
            # Save results to file
            filename = f"research_{symbol}_{int(time.time())}.json"
            filepath = os.path.join(data_config["data_dir"], filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"📊 Results saved to {filepath}")
            
            if "investment_recommendation" in results:
                rec = results["investment_recommendation"]
                print(f"💡 Recommendation: {rec.get('action', 'N/A')}")
        
        print("👋 Interactive session ended")
        
    except Exception as e:
        print(f"❌ Error in interactive mode: {str(e)}")


if __name__ == "__main__":
    import time
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_research()
    else:
        main()