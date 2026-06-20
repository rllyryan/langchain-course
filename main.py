from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List

from pydantic import BaseModel, Field

class Source(BaseModel):
    """
    Schema for a source used by the agent.
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for the agent response with the answer and sources
    """
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

llm = ChatOllama(temperature=0, model="gemma4")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 jobs posters for an ai engineer using langchain and living in Singapore."
            )
        }
    )
    print(result['structured_response'])


if __name__ == "__main__":
    main()
