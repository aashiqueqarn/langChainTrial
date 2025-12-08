from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch


load_dotenv()


tools = [TavilySearch()]

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- PROMPT ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agent that uses tools when needed."),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
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