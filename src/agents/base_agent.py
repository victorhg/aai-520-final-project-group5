"""
Base Agent class with autonomous capabilities including planning, tool usage, 
self-reflection, and memory management.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import openai
from pydantic import BaseModel


class ReflectionType(Enum):
    """Types of self-reflection the agent can perform."""
    QUALITY_ASSESSMENT = "quality_assessment"
    STRATEGY_EVALUATION = "strategy_evaluation"
    ERROR_ANALYSIS = "error_analysis"
    LEARNING_SYNTHESIS = "learning_synthesis"


@dataclass
class Memory:
    """Represents a memory entry that the agent can learn from."""
    timestamp: datetime
    context: str
    action: str
    outcome: str
    reflection: str
    quality_score: float
    tags: List[str]


@dataclass
class Plan:
    """Represents a research plan with multiple steps."""
    goal: str
    steps: List[str]
    estimated_time: int  # in minutes
    required_tools: List[str]
    success_criteria: List[str]


class Tool(ABC):
    """Abstract base class for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass


class BaseAgent(ABC):
    """
    Base Agent class providing autonomous capabilities:
    - Planning research steps
    - Dynamic tool usage
    - Self-reflection on output quality
    - Learning across runs through memory
    """
    
    def __init__(
        self, 
        name: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        memory_file: str = "./data/agent_memory.json"
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.memory_file = memory_file
        self.tools: Dict[str, Tool] = {}
        self.memory: List[Memory] = []
        self.current_plan: Optional[Plan] = None
        
        # Load existing memory
        self._load_memory()
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool
    
    def create_plan(self, goal: str, context: str = "") -> Plan:
        """
        Create a research plan for achieving the given goal.
        Uses AI to generate logical steps and identify required tools.
        """
        prompt = f"""
        As an autonomous investment research agent, create a detailed plan to achieve this goal:
        Goal: {goal}
        
        Context: {context}
        
        Available tools: {list(self.tools.keys())}
        
        Create a plan with:
        1. Logical research steps (3-7 steps)
        2. Estimated time in minutes
        3. Required tools for each step
        4. Success criteria to evaluate completion
        
        Respond in JSON format:
        {{
            "goal": "...",
            "steps": ["step1", "step2", ...],
            "estimated_time": 30,
            "required_tools": ["tool1", "tool2", ...],
            "success_criteria": ["criteria1", "criteria2", ...]
        }}
        """
        
        response = self._call_llm(prompt)
        plan_data = json.loads(response)
        
        plan = Plan(**plan_data)
        self.current_plan = plan
        return plan
    
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Execute the research plan step by step.
        Dynamically uses tools and adapts based on intermediate results.
        """
        results = {
            "plan": asdict(plan),
            "step_results": [],
            "final_analysis": "",
            "execution_time": 0
        }
        
        start_time = time.time()
        
        for i, step in enumerate(plan.steps):
            print(f"Executing step {i+1}: {step}")
            
            step_result = self._execute_step(step, results["step_results"])
            results["step_results"].append({
                "step": step,
                "result": step_result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate final analysis
        results["final_analysis"] = self._synthesize_results(results["step_results"])
        results["execution_time"] = time.time() - start_time
        
        return results
    
    def reflect_on_output(self, output: Dict[str, Any], reflection_type: ReflectionType) -> Dict[str, Any]:
        """
        Perform self-reflection on the agent's output to assess quality and identify improvements.
        """
        reflection_prompts = {
            ReflectionType.QUALITY_ASSESSMENT: """
            Assess the quality of this investment research output:
            {output}
            
            Rate on a scale of 1-10 and provide specific feedback on:
            1. Completeness of analysis
            2. Data quality and sources
            3. Logical reasoning
            4. Actionable insights
            5. Risk assessment
            
            Provide a JSON response with:
            {{
                "overall_score": 8.5,
                "strengths": ["strength1", "strength2"],
                "weaknesses": ["weakness1", "weakness2"],
                "improvement_suggestions": ["suggestion1", "suggestion2"]
            }}
            """,
            
            ReflectionType.STRATEGY_EVALUATION: """
            Evaluate the research strategy used:
            {output}
            
            Assess:
            1. Appropriateness of tools used
            2. Sequence of research steps
            3. Time allocation
            4. Information gathering effectiveness
            
            Suggest strategy improvements for future research.
            """,
            
            ReflectionType.ERROR_ANALYSIS: """
            Analyze any errors or issues in this research:
            {output}
            
            Identify:
            1. Data collection errors
            2. Analysis mistakes
            3. Logical inconsistencies
            4. Missing information
            
            Provide corrective actions for each identified issue.
            """,
            
            ReflectionType.LEARNING_SYNTHESIS: """
            Extract key learnings from this research session:
            {output}
            
            Identify:
            1. Successful strategies to repeat
            2. Patterns in market behavior
            3. Tool effectiveness insights
            4. Process improvements
            
            Synthesize into actionable knowledge for future research.
            """
        }
        
        prompt = reflection_prompts[reflection_type].format(output=json.dumps(output, indent=2))
        reflection_result = self._call_llm(prompt)
        
        try:
            reflection_data = json.loads(reflection_result)
        except json.JSONDecodeError:
            reflection_data = {"reflection": reflection_result}
        
        # Store reflection in memory
        memory_entry = Memory(
            timestamp=datetime.now(),
            context=f"Reflection on research output",
            action=f"Performed {reflection_type.value}",
            outcome=reflection_result,
            reflection=json.dumps(reflection_data),
            quality_score=reflection_data.get("overall_score", 5.0),
            tags=[reflection_type.value, "self_reflection"]
        )
        self._add_to_memory(memory_entry)
        
        return reflection_data
    
    def learn_from_memory(self, context: str = "") -> str:
        """
        Extract relevant insights from past experiences to improve current research.
        """
        if not self.memory:
            return "No previous experiences to learn from."
        
        # Get recent high-quality memories
        recent_memories = sorted(self.memory, key=lambda m: m.timestamp, reverse=True)[:10]
        high_quality_memories = [m for m in recent_memories if m.quality_score >= 7.0]
        
        memory_summaries = []
        for mem in high_quality_memories:
            memory_summaries.append({
                "context": mem.context,
                "action": mem.action,
                "outcome": mem.outcome,
                "quality_score": mem.quality_score,
                "tags": mem.tags
            })
        
        prompt = f"""
        Based on these past research experiences, provide insights for the current context:
        Current context: {context}
        
        Past experiences:
        {json.dumps(memory_summaries, indent=2)}
        
        Extract:
        1. Relevant patterns or strategies that worked well
        2. Common pitfalls to avoid
        3. Tool combinations that were effective
        4. Market insights that might apply
        
        Provide actionable recommendations for the current research.
        """
        
        insights = self._call_llm(prompt)
        return insights
    
    def _execute_step(self, step: str, previous_results: List[Dict]) -> Dict[str, Any]:
        """Execute a single step of the research plan."""
        # Determine which tools to use for this step
        tool_selection_prompt = f"""
        For this research step: "{step}"
        
        Available tools: {list(self.tools.keys())}
        Previous results: {json.dumps(previous_results[-2:], indent=2) if previous_results else "None"}
        
        Which tool should be used and with what parameters?
        Respond in JSON format:
        {{
            "tool": "tool_name",
            "parameters": {{"param1": "value1", "param2": "value2"}}
        }}
        """
        
        tool_decision = self._call_llm(tool_selection_prompt)
        
        try:
            tool_config = json.loads(tool_decision)
            tool_name = tool_config["tool"]
            parameters = tool_config.get("parameters", {})
            
            if tool_name in self.tools:
                result = self.tools[tool_name].execute(**parameters)
                return result
            else:
                return {"error": f"Tool {tool_name} not available"}
        
        except Exception as e:
            return {"error": f"Failed to execute step: {str(e)}"}
    
    def _synthesize_results(self, step_results: List[Dict]) -> str:
        """Synthesize step results into final analysis."""
        prompt = f"""
        Synthesize these research step results into a comprehensive investment analysis:
        
        {json.dumps(step_results, indent=2)}
        
        Provide:
        1. Executive summary
        2. Key findings and insights
        3. Investment recommendation with rationale
        4. Risk assessment
        5. Supporting data and evidence
        
        Make it actionable and well-structured.
        """
        
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the language model with the given prompt."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def _load_memory(self) -> None:
        """Load memory from file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    self.memory = [
                        Memory(
                            timestamp=datetime.fromisoformat(m["timestamp"]),
                            context=m["context"],
                            action=m["action"],
                            outcome=m["outcome"],
                            reflection=m["reflection"],
                            quality_score=m["quality_score"],
                            tags=m["tags"]
                        )
                        for m in memory_data
                    ]
            except Exception as e:
                print(f"Error loading memory: {e}")
                self.memory = []
    
    def _save_memory(self) -> None:
        """Save memory to file."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        memory_data = [
            {
                "timestamp": m.timestamp.isoformat(),
                "context": m.context,
                "action": m.action,
                "outcome": m.outcome,
                "reflection": m.reflection,
                "quality_score": m.quality_score,
                "tags": m.tags
            }
            for m in self.memory
        ]
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def _add_to_memory(self, memory: Memory) -> None:
        """Add a memory entry and save to file."""
        self.memory.append(memory)
        
        # Keep only the most recent 1000 memories
        if len(self.memory) > 1000:
            self.memory = sorted(self.memory, key=lambda m: m.timestamp)[-1000:]
        
        self._save_memory()
    
    @abstractmethod
    def research_stock(self, symbol: str) -> Dict[str, Any]:
        """Main research method to be implemented by subclasses."""
        pass