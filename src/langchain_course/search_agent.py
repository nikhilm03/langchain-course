from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """A simple search tool that simulates searching for information based on a query.
    In a real-world scenario, this function would interface with a search engine or database to retrieve relevant information."""
    print(f"Searching for: {query}")
    # return "Tokyo weather is very very sunny"
    return tavily.search(query=query)

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
    temperature=0,
    
)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(
    llm,
    tools=tools,    
)

def main():
    print("Hello from langchain-course!")
    # user_input = "What is the weather like in Tokyo?"
    user_input = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details."
    search_result = search.invoke(user_input)
    print("Agent Response:", search_result)


if __name__ == "__main__":
    main()