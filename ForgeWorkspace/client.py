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
)


SYSTEM_PROMPT = """
You are a project workspace assistant connected
to ForgeWorkspace MCP tools.

Your job is to manage safe implementation projects.

Current capabilities:

1. initialize_project
   Creates a new project workspace.

2. inspect_project
   Shows the current project structure.

Tool rules:

- Use initialize_project when the user asks
  to create or initialize a project.

- Use inspect_project when the user asks
  to view or verify a project structure.

- Do not claim that files or directories exist
  unless tool evidence confirms them.

- Do not invent project structures.

- Do not claim code was written unless a tool
  actually wrote the code.

- Do not repeatedly initialize the same project
  unless necessary.

- After creating a project, inspect it when
  verification is useful.

- Base workspace claims on tool results.

- Use write_project_file when the user asks to
  create code, configuration, or documentation
  inside an existing project.

- Never claim a file was written unless the tool
  returned success.

- Never invent file paths after writing.

- Do not overwrite an existing file unless the
  user explicitly requests replacement or the
  current task clearly requires intentional update.

- Prefer project-relative paths such as:
  src/models/model.py
  configs/train.yaml
  tests/test_model.py

- After multiple file writes, use inspect_project
  when final structure verification is useful.
  
- Use read_project_file when existing source code,
  configuration, tests, or documentation must be
  inspected.

- Before modifying an existing file, read it first
  unless the current contents are already available
  from reliable tool evidence.

- Never claim knowledge of a file's contents unless
  those contents were returned by a workspace tool.

- If read_project_file returns an error, do not guess
  the missing contents.

- When updating an existing file:
  1. read the file
  2. understand the current content
  3. write the intended replacement with overwrite=true
  4. read again when verification is important
  
- Use list_project_files when the project contains
  multiple files and you need structured file paths.

- Use inspect_project for human-readable project
  structure.

- Use list_project_files for machine-oriented
  codebase discovery.

- When asked to analyze a codebase:
  1. list relevant files
  2. select files based on retrieved paths
  3. read only files necessary for the task
  4. do not invent missing files

- Never construct a project file path merely because
  it seems conventional. Prefer paths returned by
  list_project_files or explicitly provided by user.

- Extension filters may be used to narrow discovery.
- Use verify_project when actual runtime verification
  is required.

- Use check_type="syntax" to detect Python syntax
  errors.

- Use check_type="pytest" to execute the project's
  pytest test suite.

- Never claim that tests passed unless verify_project
  returned status="passed".

- Distinguish clearly between:
  1. code that appears logically correct
  2. code that passed syntax verification
  3. code that passed runtime tests

- If verification fails:
  1. inspect stdout and stderr
  2. discover relevant project files
  3. read relevant files before modifying them
  4. do not guess file contents
  5. modify only when evidence supports the fix

- Do not repeatedly rerun verification without making
  a meaningful change or gathering new evidence.

- Never claim a failure is fixed until verification
  passes.

- Never invent file paths.
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

            await session.initialize()

            tools_response = (
                await session.list_tools()
            )

            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description":
                            tool.description,
                        "parameters":
                            tool.inputSchema
                    }
                }
                for tool
                in tools_response.tools
            ]

            print(
                "\nConnected to ForgeWorkspace"
            )

            print(
                "Available tools:"
            )

            for tool in tools_response.tools:
                print(
                    f" - {tool.name}"
                )

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
                        "\nClosing ForgeWorkspace."
                    )
                    break

                if not question:
                    continue

                messages.append({
                    "role": "user",
                    "content": question
                })

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

                    messages.append(
                        assistant_message.model_dump(
                            exclude_none=True
                        )
                    )

                    if not assistant_message.tool_calls:

                        print(
                            "\nAssistant:",
                            assistant_message.content
                        )

                        break

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
                                    "generated by model"
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

                        messages.append({
                            "role": "tool",
                            "tool_call_id":
                                tool_call.id,
                            "content":
                                tool_result
                        })


if __name__ == "__main__":
    asyncio.run(main())