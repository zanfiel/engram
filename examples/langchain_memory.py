"""
Engram + LangChain Integration

Use Engram as persistent memory for LangChain agents.
Memories survive across sessions and are recalled by semantic relevance.
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from engram import Engram

memory = Engram("http://localhost:4200", api_key="eg_your_key")


@tool
def remember(content: str, category: str = "general") -> str:
    """Store something important for later. Categories: task, discovery, decision, state, issue."""
    result = memory.store(content, category=category, importance=7)
    return f"Stored memory #{result['id']}"


@tool
def recall(query: str) -> str:
    """Search your memory for relevant past context."""
    results = memory.search(query, limit=5)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"[{m.category}] {m.content}" for m in results)


# Build the agent
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with persistent memory. "
     "Use 'remember' to store important facts and decisions. "
     "Use 'recall' to search your memory before answering questions."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [remember, recall], prompt)
executor = AgentExecutor(agent=agent, tools=[remember, recall])

# --- Usage ---
# Session 1:
executor.invoke({"input": "My favorite language is Rust and I use NeoVim"})
# Agent calls remember() to store this

# Session 2 (hours later):
executor.invoke({"input": "Help me set up my editor"})
# Agent calls recall("editor setup") → gets NeoVim preference back
