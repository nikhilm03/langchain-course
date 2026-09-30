from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient
import re
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """A simple search tool that simulates searching for information based on a query.
    In a real-world scenario, this function would interface with a search engine or database to retrieve relevant information."""
    print(f"Searching for: {query}")
    # return "Tokyo weather is very very sunny. Its temperature is 25 degrees Celsius."
    return tavily.search(query=query)

@tool
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    print(f"Tripling the number: {num}")
    return float(num) * 3


llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    temperature=0,
    
)
tools = [search, triple]
# tools = [TavilySearch()]
agent = create_agent(
    llm,
    tools=tools,
    system_prompt=(
            "You are a helpful assistant that can answer user questions using the tools provided. "
        ),
    # system_prompt=(
    #     "For requests to find a city's temperature and triple it, call "
    #     "temperature_and_triple with the city. Report its result without "
    #     "inventing or recalculating values."
    # ),
)

def main():
    print("Hello from langchain-course!")
    # user_input = "What is the weather like in Tokyo?"
    # user_input = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details."
    user_input = "How is the waether in Tokyo? List it and then triple the temperature."
    search_result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]}
    )
    for message in search_result["messages"]:
        if getattr(message, "tool_calls", None):
            print("Tool calls:", message.tool_calls)
    print("Agent Response:", search_result["messages"][-1].content)


if __name__ == "__main__":
    main()