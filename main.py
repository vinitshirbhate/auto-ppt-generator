import os
import asyncio
import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@canva/mcp-server"],
    env={
        "CANVA_CLIENT_ID": os.getenv("CANVA_CLIENT_ID"),
        "CANVA_CLIENT_SECRET": os.getenv("CANVA_CLIENT_SECRET"),
    }
)

async def get_tools(session: ClientSession):
    mcp_tools = await session.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        }
        for t in mcp_tools.tools
    ]

async def handle_tool_calls(session: ClientSession, response):
    results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"\nCalling tool: {block.name}")
            print(f"   Input: {block.input}")
            result = await session.call_tool(block.name, block.input)
            print(f"   Result: {result}")
            results.append({
                "tool": block.name,
                "input": block.input,
                "result": result
            })
    return results

async def run(user_prompt: str):
    print(f"\n🎨 Prompt: {user_prompt}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to Canva MCP server")

            tools = await get_tools(session)
            print(f"🔧 Available tools: {[t['name'] for t in tools]}")

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            messages = [{"role": "user", "content": user_prompt}]

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    tools=tools,
                    messages=messages
                )

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text"):
                            print(f"\n Ai: {block.text}")
                    break

                tool_results = await handle_tool_calls(session, response)

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result["result"])
                        }
                        for block, result in zip(
                            [b for b in response.content if b.type == "tool_use"],
                            tool_results
                        )
                    ]
                })

if __name__ == "__main__":
    prompt = "Create a professional LinkedIn banner with a navy blue background and bold white text saying 'Building the Future'"
    asyncio.run(run(prompt))