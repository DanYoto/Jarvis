import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp import types

from LLMAgents import MultiAgentOrchestrator, AgentState

mcp = FastMCP("Multi-Agent Orchestration")
orchestrator = MultiAgentOrchestrator()

@mcp.tool()
async def simple_query(query: str) -> str:
    """
    Execute a simple query using the simple tool agent.
    Best for straightforward questions that require basic web search or time information
    Args:
        query: the query too execute
    Returns:
        Response from the simple agent
    """
    try:
        logging.info(f"Executing simple query: {query}")
        response = orchestrator.simple_agents.chat(query)
        return f"Simple Agent Response: \n{response}"
    except Exception as e:
        logging.error(f"Error in simple query: {str(e)}")
        return f"Error executing simple query: {str(e)}"
    
@mcp.tool()
async def complex_query(query: str) -> str:
    """
    Execute a complex query using the complex task agent
    Best for multi-step analysis, research tasks, investment analysis
    market research, and comprehensive reports
    Args:
        query: the complex query or analysis request
    Returns:
        Detailed analysis response from the complex agent
    """
    try:
        logging.info(f"Executing complex analysis: {query}")
        response = orchestrator.complex_agents.chat(query)
        return f"Complex Agent Analysis:\n {response}"
    except Exception as e:
        logging.error(f"Error in complex analysis: {str(e)}")
        return f"Error Executing complex analysis: {str(e)}"
    
@mcp.tool()
async def auto_route_query(query: str) -> str:
    """
    Automatically route a query to the appropriate agent based on complexity analysis.
    
    The system will analyze the query complexity and automatically choose
    between simple and complex agents for optimal results.
    
    Args:
        query: The query to analyze and route automatically
        
    Returns:
        Response from the most appropriate agent
    """
    try:
        logging.info(f"Auto-routing query: {query}")
        response = orchestrator.chat(query)
        return f"Auto-Routed Response:\n{response}"
    except Exception as e:
        logging.error(f"Error in auto-route query: {str(e)}")
        return f"Error executing auto-route query: {str(e)}"
    
@mcp.tool()
async def analyze_query_complexity(query: str) -> str:
    """
    Analyze the complexity level of a query without executing it.
    
    This tool helps you understand how the system would route your query
    and which agent would handle it.
    
    Args:
        query: The query to analyze
        
    Returns:
        Complexity analysis result
    """
    try:
        from langchain_core.messages import HumanMessage
        
        # Create a temporary state for analysis
        temp_state = AgentState(
            messages=[HumanMessage(content=query)], 
            query=query
        )
        
        complexity_level = orchestrator.complexity_analyzer.analyze_complexity(temp_state)
        
        analysis_result = {
            "query": query,
            "complexity_level": complexity_level,
            "recommended_agent": "simple_agent" if complexity_level == "simple" else "complex_agent",
            "reasoning": {
                "simple": "Query appears to be straightforward and can be handled with basic tools",
                "complex": "Query requires multi-step analysis, planning, or comprehensive research"
            }.get(complexity_level, "Unknown complexity level")
        }
        
        return f"Complexity Analysis:\n{json.dumps(analysis_result, indent=2)}"
    except Exception as e:
        logging.error(f"Error analyzing complexity: {str(e)}")
        return f"Error analyzing complexity: {str(e)}"
    
@mcp.resource("agent://system/info")
async def get_system_info() -> str:
    """Provide detailed information about the multi-agent system"""
    return """
# Multi-Agent Orchestrator System

This system consists of multiple specialized agents that work together to handle
queries of varying complexity.

## Available Agents:

### Simple Agent
- **Purpose**: Handle straightforward queries
- **Tools**: Web search, current time
- **Best for**: Quick facts, basic searches, time queries

### Complex Agent  
- **Purpose**: Handle sophisticated analysis tasks
- **Process**: Planning → Execution → Tool calls → Summarization
- **Best for**: Investment analysis, market research, strategic planning

### Complexity Analyzer
- **Purpose**: Route queries to the appropriate agent
- **Function**: Analyzes query complexity automatically

## Usage Recommendations:

- Use `auto_route_query` when unsure about complexity
- Use `simple_query` for basic information needs
- Use `complex_analysis` for detailed research and analysis
- Use `analyze_query_complexity` to understand routing decisions
    """

def main():
    """Main entry point for the FastMCP server"""
    try:
        logging.info("Starting Multi-Agent FastMCP Server...")
        mcp.run()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        raise

if __name__ == "__main__":
    main()