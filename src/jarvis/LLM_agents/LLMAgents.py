from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_openai.chat_models.azure import AzureChatOpenAI
from AgentTools.Agenttools import *
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, TypedDict, Annotated
from agent_prompts import complexity_analyze_prompt, planning_prompt, step_wise_execution_prompt

load_dotenv()

# add complexity analyzer
class ComplexityLevel:
    "Complexity levels for the agent's tasks."

    SIMPLE = "simple"
    COMPLEX = "complex"


class AgentState(TypedDict):
    """
    AgentState is a type dictionary that holds the state of the agent.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    complexity_level: Optional[ComplexityLevel]
    general_planning: Optional[str]
    intermediate_step: Optional[List[str]]


class ComplexityAnalyzer:
    """
    ComplexityAnalyzer is a class that analyzes the complexity of the agent's tasks.
    It can be used to determine the complexity level of the tasks based on the messages.
    """

    def __init__(
        self, 
        api_key: str = os.getenv("llm_api_key"), 
        deployment_name: str = "gpt-4o",
        endpoint: str = os.getenv("endpoint"),
        api_version: str = "2024-12-01-preview"
    ):
        """
        Initialize the ComplexityAnalyzer with an optional API key.
        """
        self.llm_client = AzureChatOpenAI(
            azure_deployment=deployment_name,
            azure_endpoint=endpoint,
            api_version=api_version,
            temperature=0.7,
            max_tokens=128,
            api_key=api_key,
        )
    
    def analyze_complexity(self, state: AgentState):
        user_query = complexity_analyze_prompt.format(query=state["messages"])
        response = self.llm_client.invoke(user_query)
        # more strick flow for handling unexpected extra messages
        if "simple" in response.content.lower():
            return ComplexityLevel.SIMPLE.value
        else:
            return ComplexityLevel.COMPLEX.value

class BaseLLMAgent:
    """
    BaseLLMAgent is a base class for LLM agents.
    It provides a common interface for LLM agents to interact with the LLM client.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None,
        deployment_name: str = "gpt-4o",
        endpoint: str = os.getenv("endpoint"),
        api_version: str = "2024-12-01-preview",
        temperature: float = 0.5,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or os.getenv("llm_api_key")
        if not self.api_key:
            raise ValueError("API key is required for BaseLLMAgent.")
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialize_client()
        self.tools = self._initialize_tools()
        self.base_tool_node = ToolNode(self.tools)
    
    def _initialize_client(self):
        self.llm_client = AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )

    def _initialize_tools(self):
        return [search_web, get_current_time]


class SimpleToolAgent(BaseLLMAgent):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize  the LLMClient with an optional API key.
        """
        # model initialization
        super().__init__(**kwargs)
        self._setup_graph()
    
    def _setup_graph(self):
        base_tool_node = ToolNode(self.tools)

        def detect_tool_calls(state: AgentState) -> dict:
            """
            Detect which tool calls were made in the last message.
            """
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for call in last_message.tool_calls:
                    print(
                        f"detected tool call: {call.get('name')} with args {call.get('args')}"
                    )

            result = base_tool_node.invoke(state)
            tool_messages = [
                msg for msg in result["messages"] if isinstance(msg, ToolMessage)
            ]
            return {"messages": tool_messages}

        self.graph = StateGraph(AgentState)
        self.graph.add_node("llm", self._call_llm)
        self.graph.add_node("tool", detect_tool_calls)
        self.graph.add_edge(START, "llm")
        self.graph.add_conditional_edges("llm", self._should_continue)
        self.graph.add_edge("tool", "llm")
        self.runnable = self.graph.compile()

    def _initialize_client(self):
        """
        Initialize multiple LLM clients for different purposes.
        """
        super()._initialize_client()
        self.llm_client = self.llm_client.bind_tools(self.tools)

    def _call_llm(self, state: AgentState) -> dict:
        """
        Call the LLM with the provided messages.
        """
        response = self.llm_client.invoke(state["messages"])

        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """
        Determine if the agent should continue based on the last message.
        """
        if state["messages"][-1].tool_calls:
            return "tool"
        return END

    def chat(self, text: str):
        init_state = AgentState(messages=[HumanMessage(content=text)])
        final_state = self.runnable.invoke(init_state)
        ai_messages = [
            msg for msg in final_state["messages"] if isinstance(msg, AIMessage)
        ]
        return ai_messages[-1].content

class ComplexTaskAgent(BaseLLMAgent):
    """
    ComplexTaskAgent is a class that handles complex tasks defined by the complexity analyzer.
    If the task is complex, it will go to planning, reasoning, and execution phases, which are separate clients defined in the class.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tools_description = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        self._initialize_clients()
        self._setup_graph()
    
    def _initialize_clients(self):
        """
        Initialize multiple LLM clients for different purposes.
        """
        super()._initialize_client()
        self.llm_client_with_tools = self.llm_client.bind_tools(self.tools)

    def _setup_graph(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("plan", self._create_plan)
        self.graph.add_node("execute", self._execute_step)
        self.graph.add_node("tool_call", self._detedct_tool_calls)
        self.graph.add_node("summarize", self._summarize_results)

        self.graph.add_edge(START, "plan")
        self.graph.add_edge("plan", "execute")
        self.graph.add_conditional_edges("execute", self._route_after_execution)
        self.graph.add_conditional_edges("tool_call", "execute")
        self.graph.add_edge("summarize", END)

        self.runnable = self.graph.compile()

    def _create_plan(self, state: AgentState) -> dict:
        """
        Create a plan for the complex task.
        """
        # Non-tool LLM client is used for planning
        messages = state['messages']
        query = messages[-1].content
        planning_query = planning_prompt.format(task=query)
        response = self.llm_client.invoke([HumanMessage(content=planning_query)])
        return {"messages": [response],
                "general_planning": response.content}
        
    def _execute_step(self, state: AgentState) -> dict:
        """
        Execute the next step based on the current plan and progress.
        """
        # tool-based LLM client is used for step-wise execution
        general_planning = state.get("general_planning", "")
        intermediate_steps = state.get("intermediate_step", [])
        execution_query = step_wise_execution_prompt.format(
            general_planning=general_planning,
            tools=self.tools_description,
            intermediate_results=intermediate_steps
        )
        response = self.llm_client_with_tools.invoke([HumanMessage(content=execution_query)])
        intermediate_steps.append(response.content)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                print(
                    f"detected tool call: {call.get('name')} with args {call.get('args')}"
                )
        
        result = self.base_tool_node.invoke(state)
        tool_messages = [
            msg for msg in result["messages"] if isinstance(msg, ToolMessage)
        ]
        if tool_messages:
            latest = tool_messages[-1]
            print(f"Tool call detected: {latest.name} with content {latest.content}")
        

        # intermediate steps is a list, is the way of updating the same as str?
        intermediate_steps.append(latest)
        raise {"messages": [response.content + latest],
               "intermediate_step": intermediate_steps}
    
    def _route_after_execution(self, state: AgentState) -> dict:
        """
        Route the evaluation steps based on the complexity of the task.
        """
        routing_query = routing_prompt.format(
            general_planning = state.get("general_planning", ""),
            tools_description = self.tools_description,
            intermediate_results = state.get("intermediate_step", []),
        )
        response = self.llm_client.invoke([HumanMessage(content=routing_query)])
        decision = response.content.strip().lower()
        if decision == "complete":
            return "summarize"
        return "execute"


class MultiAgentOrchestrator:
    """
    MultiAgentOrchestrator is a class that orchestrates multiple agents to handle complex tasks.
    It can be used to manage the flow of tasks between different agents based on their capabilities.
    """

    def __init__(
        self, 
        simple_agents: Optional[SimpleToolAgent] = None, 
        complex_agents: Optional[ComplexTaskAgent] = None,
        complexity_analyzer: Optional[ComplexityAnalyzer] = None
    ):
        self.simple_agents = simple_agents or SimpleToolAgent()
        self.complex_agents = complex_agents or ComplexTaskAgent()
        self.complexity_analyzer = complexity_analyzer or ComplexityAnalyzer()
        self._setup_orchestrator_graph()
    
    def _setup_orchestrator_graph(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("analyze_complexity", self._analyze_complexity)
        self.graph.add_node("simple_agent", self._run_simple_agent)
        self.graph.add_node("complex_agent", self._run_complex_agent)

        self.graph.add_edge(START, "analyze_complexity")
        self.graph.add_conditional_edges("analyze_complexity", self._route_to_agent)
        self.graph.add_edge("simple_agent", END)
        self.graph.add_edge("complex_agent", END)

    def _analyze_complexity(self, state: AgentState) -> dict:
        """
        Analyze the complexity of the task using the ComplexityAnalyzer.
        """
        complexity_level = self.complexity_analyzer.analyze_complexity(state)
        return {"complexity_level": complexity_level}

    def _route_to_agent(self, state: AgentState) -> str:
        """
        Route the task to the appropriate agent based on the complexity level.
        """
        if state["complexity_level"] == ComplexityLevel.SIMPLE.value:
            return "simple_agent"
        else:
            return "complex_agent"
    
    def _run_simple_agent(self, state: AgentState) -> dict:
        """
        Run the simple agent to handle simple tasks.
        """
        response = self.simple_agents.chat(state["messages"][-1].content)
        return {"messages": [AIMessage(content=response)]}

    def _run_complex_agent(self, state: AgentState) -> dict:
        """
        Run the complex agent to handle complex tasks.
        """
        response = self.complex_agents.chat(state["messages"][-1].content)
        return {"messages": [AIMessage(content=response)]}

if __name__ == "__main__":
    agent = LLMClient(api_key=os.getenv("llm_api_key"), deployment_name="gpt-4o")
    final_response = agent.chat(
        "As of June 1, 2023, which three films have earned the highest worldwide box-office grosses for 2023 so far?"
    )
    print(final_response)
