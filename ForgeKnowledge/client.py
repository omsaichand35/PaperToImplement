import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

from paperforge_env import load_project_env

load_project_env()

from mcp.client.session import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters
)


client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
)

MAX_RETRIEVAL_CALLS = 3


async def main():

    # Locate server.py
    server_path = os.path.join(
        os.path.dirname(__file__),
        "server.py"
    )

    params = StdioServerParameters(
        command=sys.executable,
        args=[server_path]
    )

    async with stdio_client(params) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            # 1. Initialize MCP connection
            await session.initialize()

            # 2. Discover tools from ForgeKnowledge
            tools_response = await session.list_tools()

            # 3. Convert MCP tools
            #    into OpenAI-compatible format
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tools_response.tools
            ]

            print("\nConnected to ForgeKnowledge")
            print("Available tools:")

            for tool in tools_response.tools:
                print(f" - {tool.name}")

            # Conversation memory
            messages = [
                {
                    "role": "system",
                    "content": """
You are a PyTorch knowledge assistant connected
to ForgeKnowledge.

ForgeKnowledge contains indexed trusted
documentation.

Rules:

1. For PyTorch factual claims, prefer retrieved
   evidence over model memory.

2. If the exact API symbol is known, call
   get_api_reference first.

3. If the user asks about one specific parameter,
   prefer get_parameter_reference.

4. Use search_docs only when:
   - the exact API symbol is unknown
   - the question is conceptual
   - discovery is required

5. Do not repeatedly search for the same fact
   using minor query variations.

6. Maximum recommended retrieval calls per
   question: 3.

7. Stop retrieving once sufficient evidence
   has been found.

8. Never claim that a fact came from retrieved
   documentation unless the tool result actually
   supports it.

9. If you add information from general model
   knowledge, label it clearly as not retrieved
   from ForgeKnowledge.

10. When evidence is insufficient, say so.
"""
                }
            ]

            # Interactive chat loop
            while True:

                question = await asyncio.to_thread(
                    input,
                    "\nYou: "
                )

                if question.lower().strip() in {
                    "exit",
                    "quit"
                }:
                    break

                messages.append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

                retrieval_calls = 0

                # Agent tool loop
                while True:

                    response = (
                        await client
                        .chat
                        .completions
                        .create(
                            model=(
                                "nvidia/"
                                "nemotron-3-nano-omni-30b-a3b-reasoning"
                            ),
                            messages=messages,
                            tools=openai_tools
                        )
                    )

                    assistant_message = (
                        response
                        .choices[0]
                        .message
                    )

                    # Save assistant response
                    messages.append(
                        assistant_message.model_dump(
                            exclude_none=True
                        )
                    )

                    # If no tool call:
                    # final answer reached
                    if not assistant_message.tool_calls:

                        print(
                            "\nAssistant:",
                            assistant_message.content
                        )

                        break

                    if retrieval_calls >= MAX_RETRIEVAL_CALLS:

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": (
                                    assistant_message
                                    .tool_calls[0]
                                    .id
                                ),
                                "content": (
                                    "ForgeKnowledge retrieval budget "
                                    "reached for this question. "
                                    "Use existing evidence or state "
                                    "that evidence is insufficient."
                                )
                            }
                        )

                        continue

                    # Execute requested MCP tools
                    for tool_call in (
                        assistant_message.tool_calls
                    ):

                        if retrieval_calls >= MAX_RETRIEVAL_CALLS:
                            break

                        tool_name = (
                            tool_call
                            .function
                            .name
                        )

                        tool_args = json.loads(
                            tool_call
                            .function
                            .arguments
                        )

                        print(
                            f"\n[Tool Call] {tool_name}"
                        )

                        print(
                            f"[Arguments] {tool_args}"
                        )

                        # Call ForgeKnowledge MCP
                        result = await session.call_tool(
                            tool_name,
                            tool_args
                        )

                        # Convert MCP content
                        # into plain text
                        tool_result = "\n".join(
                            content.text
                            for content in result.content
                            if hasattr(content, "text")
                        )

                        print(
                            f"[Tool Result]\n"
                            f"{tool_result}"
                        )

                        # Critical:
                        # return evidence to LLM
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            }
                        )

                        retrieval_calls += 1


if __name__ == "__main__":
    asyncio.run(main())
