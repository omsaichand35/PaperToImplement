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
    base_url=(
        "https://integrate.api.nvidia.com/v1"
    ),
    max_retries=2,
)


SYSTEM_PROMPT = """
You are a code review assistant connected to
the ForgeCode MCP server.

Your job is to review generated implementation
artifacts against a specification and plan using
the review_implementation_tool.

Rules:

1. Call review_implementation_tool with:
   - spec: the original specification (plain text)
   - plan: the validated implementation plan
           as a JSON string
   - artifacts: the generated file artifacts
                as a JSON array string

2. Interpret the review result for the user:
   - passed: whether the implementation is accepted
   - issues: list of detected problems with
             severity, category, evidence, and
             affected files
   - missing_requirements: spec requirements
     absent from the implementation
   - invented_details: architectural details
     added without spec support
   - cross_file_inconsistencies: mismatches
     between generated files

3. Do not invent review outcomes.

4. If the tool returns an error, report it
   clearly and do not fabricate a review.

5. Stop calling tools once the review is
   complete.

6. Present the review in a clear, structured
   summary for the user.
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
                "\nConnected to ForgeCode"
            )

            print(
                "Available tools:"
            )

            for tool in tools_response.tools:
                print(
                    f" - {tool.name}"
                )

            print(
                "\nPaste your spec, plan JSON, "
                "and artifacts JSON.\n"
                "Type 'exit' or 'quit' to stop.\n"
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
                        "\nClosing ForgeCode."
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
                            model="meta/llama-3.1-70b-instruct",
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
