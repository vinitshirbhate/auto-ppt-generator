import os
import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import json

load_dotenv()

# ── MCP server config ──────────────────────────────────────────────
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@canva/mcp-server"],
    env={
        "CANVA_CLIENT_ID": os.getenv("CANVA_CLIENT_ID"),
        "CANVA_CLIENT_SECRET": os.getenv("CANVA_CLIENT_SECRET"),
    }
)

# ── LangGraph state ────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ── Main agent ─────────────────────────────────────────────────────
async def run(user_prompt: str):
    print(f"\n Prompt: {user_prompt}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to Canva MCP server")

            # Fetch MCP tools and wrap as LangChain tools
            mcp_tools_list = await session.list_tools()
            print(f"🔧 Available tools: {[t.name for t in mcp_tools_list.tools]}")

            lc_tools = []
            for t in mcp_tools_list.tools:
                async def make_tool_fn(tool_name):
                    async def tool_fn(**kwargs):
                        print(f"\ Calling tool: {tool_name}")
                        print(f"   Input: {kwargs}")
                        result = await session.call_tool(tool_name, kwargs)
                        print(f"   Result: {result}")
                        return str(result)
                    return tool_fn

                fn = await make_tool_fn(t.name)
                lc_tools.append(
                    StructuredTool.from_function(
                        coroutine=fn,
                        name=t.name,
                        description=t.description or "",
                        args_schema=None,
                    )
                )

            # OpenRouter LLM
            llm = ChatOpenAI(
                model="anthropic/claude-3.5-sonnet",   # or any OpenRouter model
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
            ).bind_tools(lc_tools)

            tool_map = {t.name: t for t in lc_tools}

            # ── Graph nodes ────────────────────────────────────────
            async def call_llm(state: AgentState):
                response = await llm.ainvoke(state["messages"])
                return {"messages": [response]}

            async def call_tools(state: AgentState):
                last_msg = state["messages"][-1]
                tool_messages = []
                for tool_call in last_msg.tool_calls:
                    tool = tool_map.get(tool_call["name"])
                    if tool:
                        result = await tool.ainvoke(tool_call["args"])
                        tool_messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"]
                            )
                        )
                return {"messages": tool_messages}

            def should_continue(state: AgentState):
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    return "tools"
                return END

            # ── Build graph ────────────────────────────────────────
            graph = StateGraph(AgentState)
            graph.add_node("llm", call_llm)
            graph.add_node("tools", call_tools)
            graph.set_entry_point("llm")
            graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
            graph.add_edge("tools", "llm")
            app = graph.compile()

            # ── Run ────────────────────────────────────────────────
            result = await app.ainvoke({"messages": [HumanMessage(content=user_prompt)]})

            final = result["messages"][-1]
            print(f"\AI: {final.content}")

if __name__ == "__main__":
    prompt = "Create a professional LinkedIn banner with a navy blue background and bold white text saying 'Building the Future'"
    asyncio.run(run(prompt))