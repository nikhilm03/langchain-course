from dotenv import load_dotenv
from langgraph import graph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from langchain_course.langgraph.react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are an assistant that must use tools to answer questions involving current information or calculations."""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)