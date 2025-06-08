from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_openai.chat_models.azure import AzureChatOpenAI
import os
from typing import Optional, List, Dict, TypedDict, Annotated
import time

class AgentState(TypedDict):
    """
    AgentState is a type dictionary that holds the state of the agent.
    """
    messages: Annotated[list[AnyMessage], add_messages]


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, deployment_name: str = "gpt-4o", endpoint: str = os.getenv("endpoint"), api_version: str = "2024-12-01-preview"):
        """
        Initialize  the LLMClient with an optional API key.
        """
        # model initialization
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for LLMClient.")
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        self.api_version = api_version

        base_tool_node = ToolNode(self._initialize_tools())
        self._initialize_client()

        def detect_tool_calls(state: AgentState) -> dict:
            """
            Detect which tool calls were made in the last message.
            """
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for call in last_message.tool_calls:
                    print(f"detected tool call: {call.get('name')} with args {call.get('args')}")

            result = base_tool_node.invoke(state)
            tool_messages = [msg for msg in result['messages'] if isinstance(msg, ToolMessage)]
            if tool_messages:
                latest = tool_messages[-1]
                print(f"Tool call detected: {latest.name} with content {latest.content}")
            return {"messages": tool_messages}

        self.graph = StateGraph(AgentState)
        self.graph.add_node('llm', self._call_llm)
        self.graph.add_node('tool', detect_tool_calls)
        self.graph.add_edge(START, 'llm')
        self.graph.add_conditional_edges('llm', self._should_continue)
        self.graph.add_edge('tool', 'llm')

        self.runnable = self.graph.compile()

    def _initialize_tools(self):
        """
        Initialize the tools for the agent.
        """
        @tool
        def search(query: str):
            """
            Search the web using Tavily.
            """
            tavily_tool = TavilySearchResults(tavily_api_key=os.environ["tavily_api_key"], max_results=5)
            results = tavily_tool.invoke(query)
            return results
        return [search]


    def _initialize_client(self):
        """
        Initialize the LLM client with the provided API key and model name.
        """
        self.llm_client = AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            temperature=0.5,
            max_tokens=2000,
            api_key=self.api_key
        )
        self.llm_client = self.llm_client.bind_tools(self._initialize_tools())


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
            return 'tool'
        return END
    

    def chat(self, text: str):
        init_state = AgentState(messages=[HumanMessage(content=text)])
        final_state = self.runnable.invoke(init_state)
        ai_messages = [msg for msg in final_state['messages'] if isinstance(msg, AIMessage)]
        return ai_messages[-1]


if __name__ == "__main__":
    agent = LLMClient(api_key=os.getenv("llm_api_key"), deployment_name="gpt-4o")
    final_response = agent.chat("As of June 1, 2023, which three films have earned the highest worldwide box-office grosses for 2023 so far?")
    print(final_response)