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


SYSTEM_PROMPT = """
You are a research investigation assistant connected
to ForgeResearch MCP tools.

Your job is to discover and inspect public
research resources using retrieved evidence.

Tool selection rules:

1. Use find_research_paper when:
   - the user provides a paper title
   - no known research page URL is available
   - paper discovery is required

2. Use inspect_research_page when:
   - the user provides a public research page URL
   - a discovered landing-page URL should be inspected
   - useful links should be extracted from a known page

3. You may chain tools when justified.

Example:

User asks to find resources for a paper title.

First:
find_research_paper(title)

Then, if a strong candidate contains a usable
HTML landing-page URL:
inspect_research_page(url)

Evidence rules:

4. Never invent URLs.

5. When a discovered candidate includes
    urls.research_landing_page, inspect that URL
    first.

6. Never inspect urls.provider_record as the
    paper landing page unless explicitly asked.

7. urls.provider_record is provenance metadata
    only.

8. A discovery candidate is not automatically
    the exact intended paper.

9. An exact title match does not prove the
    candidate is canonical or original.

10. Use publication_year, authors, doi, venue,
     and source provenance as identity evidence.

11. If metadata conflicts, report the conflict
     instead of resolving it from memory.

12. Stop calling tools once sufficient evidence
     has been retrieved.

13. Do not repeatedly inspect the same URL.

14. Clearly distinguish:
    - discovered candidate
    - inspected page
    - paper PDF
    - repository
    - official repository
    - supplementary material
    - dataset candidate

15. If evidence is insufficient, say so clearly.
"""


async def main():

    server_path = os.path.join(
        os.path.dirname(__file__),
        "server.py"
    )

    params = StdioServerParameters(
        command=sys.executable,
        args=[server_path]
    )

    async with stdio_client(
        params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            # Initialize MCP connection
            await session.initialize()

            # Discover MCP tools
            tools_response = (
                await session.list_tools()
            )

            # Convert MCP tools into
            # OpenAI-compatible function tools
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": (
                            tool.description
                        ),
                        "parameters": (
                            tool.inputSchema
                        )
                    }
                }
                for tool in tools_response.tools
            ]

            print(
                "\nConnected to ForgeResearch"
            )

            print(
                "Available tools:"
            )

            for tool in tools_response.tools:
                print(
                    f" - {tool.name}"
                )

            # Conversation memory
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            while True:

                question = await asyncio.to_thread(
                    input,
                    "\nYou: "
                )

                question = question.strip()

                if question.lower() in {
                    "exit",
                    "quit"
                }:
                    print(
                        "\nClosing ForgeResearch."
                    )
                    break

                if not question:
                    continue

                messages.append({
                    "role": "user",
                    "content": question
                })

                # Agent tool loop
                while True:

                    response = await (
                        client
                        .chat
                        .completions
                        .create(
                            model=(
                                "nvidia/"
                                "nemotron-3-nano-omni-"
                                "30b-a3b-reasoning"
                            ),
                            messages=messages,
                            tools=openai_tools,
                            temperature=0.1
                        )
                    )

                    assistant_message = (
                        response
                        .choices[0]
                        .message
                    )

                    # Preserve assistant message,
                    # including tool calls
                    messages.append(
                        assistant_message.model_dump(
                            exclude_none=True
                        )
                    )

                    # No tool calls means:
                    # final answer reached
                    if not assistant_message.tool_calls:

                        print(
                            "\nAssistant:",
                            assistant_message.content
                        )

                        break

                    # Execute all requested tools
                    for tool_call in (
                        assistant_message.tool_calls
                    ):

                        tool_name = (
                            tool_call
                            .function
                            .name
                        )

                        try:
                            tool_args = json.loads(
                                tool_call
                                .function
                                .arguments
                            )

                        except json.JSONDecodeError:

                            tool_result = json.dumps({
                                "status": "error",
                                "error": (
                                    "Invalid JSON arguments "
                                    "generated by the model"
                                )
                            })

                            messages.append({
                                "role": "tool",
                                "tool_call_id":
                                    tool_call.id,
                                "content":
                                    tool_result
                            })

                            continue

                        print(
                            f"\n[Tool Call] "
                            f"{tool_name}"
                        )

                        print(
                            f"[Arguments] "
                            f"{tool_args}"
                        )

                        try:

                            result = (
                                await session.call_tool(
                                    tool_name,
                                    tool_args
                                )
                            )

                            tool_result = "\n".join(
                                content.text
                                for content
                                in result.content
                                if hasattr(
                                    content,
                                    "text"
                                )
                            )

                            if not tool_result:
                                tool_result = json.dumps({
                                    "status": "error",
                                    "error": (
                                        "Tool returned "
                                        "no text content"
                                    )
                                })

                        except Exception as error:

                            tool_result = json.dumps({
                                "status": "error",
                                "tool": tool_name,
                                "error": str(error)
                            })

                        print(
                            f"[Tool Result]\n"
                            f"{tool_result}"
                        )

                        # Critical:
                        # send tool evidence back
                        # to the LLM
                        messages.append({
                            "role": "tool",
                            "tool_call_id":
                                tool_call.id,
                            "content":
                                tool_result
                        })


if __name__ == "__main__":
    asyncio.run(main())