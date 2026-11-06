from langgraph.graph import START,END,StateGraph
from state import ChatState
from  nodes  import generate_node
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from tools import tools


builder = StateGraph(ChatState)

conn= sqlite3.connect(database='qwen_assistant.db',check_same_thread=False)
checkpointer= SqliteSaver(conn=conn)

builder.add_node("generate", generate_node)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"generate")
builder.add_conditional_edges(
    "generate",
    tools_condition,
    {
        "tools":"tools",
        "__end__":END
    }
)
builder.add_edge("tools", "generate")

graph= builder.compile(checkpointer=checkpointer)

