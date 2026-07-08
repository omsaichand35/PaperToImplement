import json

from mcp.server.fastmcp import FastMCP

from codeforge.reviewer import review_implementation


mcp = FastMCP(
    "ForgeCode"
)


def json_response(
    payload: dict
) -> str:

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False
    )


@mcp.tool()
async def review_implementation_tool(
    spec: str,
    plan: str,
    artifacts: str
) -> str:
    """
    Review generated project artifacts against
    the original specification and validated plan.

    Arguments:
        spec      -- the original implementation
                     specification (plain text)
        plan      -- the validated implementation plan
                     as a JSON string
        artifacts -- the generated file artifacts
                     as a JSON array string

    Returns a structured review indicating whether
    the implementation passed, along with any issues,
    missing requirements, invented details, or
    cross-file inconsistencies detected.

    Use this tool after generate_file_tool has
    produced all planned artifacts.
    """

    if not spec.strip():
        return json_response({
            "status": "error",
            "error": "spec cannot be empty"
        })

    if not plan.strip():
        return json_response({
            "status": "error",
            "error": "plan cannot be empty"
        })

    if not artifacts.strip():
        return json_response({
            "status": "error",
            "error": "artifacts cannot be empty"
        })

    try:
        parsed_plan = json.loads(plan)

    except json.JSONDecodeError as error:
        return json_response({
            "status": "error",
            "error": (
                "plan is not valid JSON: "
                f"{error}"
            )
        })

    try:
        parsed_artifacts = json.loads(artifacts)

    except json.JSONDecodeError as error:
        return json_response({
            "status": "error",
            "error": (
                "artifacts is not valid JSON: "
                f"{error}"
            )
        })

    if not isinstance(parsed_artifacts, list):
        return json_response({
            "status": "error",
            "error": (
                "artifacts must be a JSON array"
            )
        })

    try:
        review = await review_implementation(
            spec=spec,
            plan=parsed_plan,
            artifacts=parsed_artifacts
        )

        return json_response({
            "status": "success",
            "review": review
        })

    except ValueError as error:
        return json_response({
            "status": "error",
            "error": str(error)
        })

    except Exception as error:
        return json_response({
            "status": "error",
            "error": (
                "Unexpected reviewer error: "
                f"{error}"
            )
        })


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )
