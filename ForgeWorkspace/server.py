import json

from mcp.server.fastmcp import FastMCP

from workspace.project import (
    initialize_project as initialize_project_impl
)

from workspace.inspector import (
    inspect_project as inspect_project_impl,
    list_project_files as list_project_files_impl
)

from workspace.files import (
    write_project_file as write_project_file_impl,
    read_project_file as read_project_file_impl
)

from workspace.verifer import (
    verify_project as verify_project_impl,
)



mcp = FastMCP(
    "ForgeWorkspace"
)


def json_response(
    payload: dict
) -> str:
    """
    Convert Python dictionaries into
    formatted JSON strings for MCP responses.
    """

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False
    )

@mcp.tool()
def verify_project(
    project_name: str,
    check_type: str = "syntax",
    timeout: int = 30
) -> str:
    """
    Run an approved verification check
    inside an existing ForgeWorkspace project.

    Supported check types:

    - syntax
      Compiles Python files to detect syntax errors.

    - pytest
      Runs the project's pytest test suite.

    This tool does not accept arbitrary commands.
    """

    try:
        result = verify_project_impl(
            project_name=project_name,
            check_type=check_type,
            timeout=timeout
        )

        return json_response(
            result
        )

    except Exception as error:
        return json_response({
            "status": "error",
            "project_name": project_name,
            "check_type": check_type,
            "error": str(error)
        })

@mcp.tool()
def initialize_project(
    project_name: str,
    framework: str = "pytorch",
    task_type: str = "deep_learning"
) -> str:
    """
    Initialize a safe implementation project
    inside the ForgeWorkspace projects directory.

    Use this tool when:
    - a new implementation project is needed
    - a paper reproduction workspace is needed
    - a project skeleton should be created

    Existing projects are not deleted.
    """

    try:

        result = initialize_project_impl(
            project_name=project_name,
            framework=framework,
            task_type=task_type
        )

        return json_response(
            result
        )

    except Exception as error:

        return json_response({
            "status": "error",
            "error": str(error)
        })


@mcp.tool()
def inspect_project(
    project_name: str
) -> str:
    """
    Inspect the current structure of an
    existing ForgeWorkspace project.

    Returns the project directory tree.
    """

    try:

        result = inspect_project_impl(
            project_name
        )

        return json_response(
            result
        )

    except Exception as error:

        return json_response({
            "status": "error",
            "project_name": project_name,
            "error": str(error)
        })

@mcp.tool()
def write_project_file(
    project_name: str,
    relative_path: str,
    content: str,
    overwrite: bool = False
) -> str:
    """
    Write a UTF-8 text file inside an
    existing ForgeWorkspace project.

    Use this tool when:
    - generated code should be saved
    - configuration files should be created
    - project documentation should be written

    The path must remain inside the project.

    Existing files are protected unless
    overwrite is explicitly true.
    """

    try:

        result = write_project_file_impl(
            project_name=project_name,
            relative_path=relative_path,
            content=content,
            overwrite=overwrite
        )

        return json_response(
            result
        )

    except Exception as error:

        return json_response({
            "status": "error",
            "project_name": project_name,
            "relative_path": relative_path,
            "error": str(error)
        })

@mcp.tool()
def read_project_file(
    project_name: str,
    relative_path: str
) -> str:
    """
    Read a UTF-8 text file from an existing
    ForgeWorkspace project.

    Use this tool when:
    - existing source code must be inspected
    - configuration should be reviewed
    - tests should be examined
    - a file must be understood before modification

    The path must remain inside the project.
    """

    try:

        result = read_project_file_impl(
            project_name=project_name,
            relative_path=relative_path
        )

        return json_response(
            result
        )

    except Exception as error:

        return json_response({
            "status": "error",
            "project_name": project_name,
            "relative_path": relative_path,
            "error": str(error)
        })

@mcp.tool()
def list_project_files(
    project_name: str,
    extensions: list[str] | None = None
) -> str:
    """
    List structured file paths inside an
    existing ForgeWorkspace project.

    Use this tool when:
    - the agent needs to discover project files
    - relevant source files must be selected
    - a codebase must be explored
    - files should be filtered by extension

    Examples of extensions:
    - ["py"]
    - [".py"]
    - ["py", "json"]
    """

    try:

        result = list_project_files_impl(
            project_name=project_name,
            extensions=extensions
        )

        return json_response(
            result
        )

    except Exception as error:

        return json_response({
            "status": "error",
            "project_name": project_name,
            "error": str(error)
        })


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )