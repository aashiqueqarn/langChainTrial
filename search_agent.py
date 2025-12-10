from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch


load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent."""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


tools = [TavilySearch()]

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, response_format=AgentResponse)

structured_llm = llm.with_structured_output(AgentResponse)

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
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    result = agent_executor.invoke({"input": "Search for 3 job postings for an ai engineer using langchain in the bay area on the linked and list their title with salary and link to the posting."})
    print("\n=== Final Output ===")
    print(result["output"])

if __name__ == "__main__":
    main()