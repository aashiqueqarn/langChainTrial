from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool

import requests

load_dotenv()

# ---- TOOL ----
def search_weather(query: str) -> str:
    """Fetch weather for a city using a free public API."""
    url = f"https://wttr.in/{query}?format=3"
    response = requests.get(url)
    return response.text

tools = [
    Tool(
        name="weather_search",
        func=search_weather,
        description="Get current weather for a city. Input should be a city name."
    )
]

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- PROMPT ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agent that uses tools when needed."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ---- AGENT ----
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    result = agent_executor.invoke({"input": "What is the weather in Tokyo?"})
    print("\n=== Final Output ===")
    print(result["output"])

if __name__ == "__main__":
    main()