from dotenv import load_dotenv
from langchain_core.tools import tool
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
    return "Tokyo weather is very very sunny. The temperature is 25 degrees Celsius."
    # return tavily.search(query=query)

@tool
def city_details(query: str) -> str:
    """A simple tool that simulates fetching details about a city based on a query."""
    print(f"Fetching details for: {query}")
    return "Tokyo is the capital of Japan, known for its modern architecture, shopping, and culture."

@tool
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    print(f"Tripling the number: {num}")
    return float(num) * 3

tools = [search, city_details,triple]

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    temperature=0,
    
).bind_tools(tools)