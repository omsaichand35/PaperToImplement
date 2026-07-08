import json

from mcp.server.fastmcp import FastMCP

from research.inspector import (
    inspect_research_page as inspect_page
)
from research.discovery import (
    find_research_papers
)

mcp = FastMCP(
    "ForgeResearch"
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
def inspect_research_page(
    url: str
) -> str:
    """
    Inspect a public research-related HTML page.

    Extracts:
    - page metadata
    - normalized links
    - repository candidates
    - paper PDF links
    - supplementary material links
    - dataset candidates
    - code candidates

    Use this tool when a known research page URL
    should be inspected for useful resources.

    For discovered candidates, prefer
    urls.research_landing_page. Do not use
    urls.provider_record as the page URL unless
    the user explicitly asks to inspect the
    provider record.
    """

    if not url.strip():
        return json_response({
            "status": "error",
            "error": "url cannot be empty"
        })

    try:
        result = inspect_page(
            url
        )

        return json_response(
            result
        )

    except Exception as error:

        return json_response({
            "status": "error",
            "url": url,
            "error": str(error)
        })

@mcp.tool()
def find_research_paper(
    title: str,
    limit: int = 5
) -> str:
    """
    Find candidate scholarly works by paper title.

    Use this tool when:
    - the user knows a paper title
    - no research page URL is available
    - a paper needs to be discovered before inspection

    Returns ranked candidates.

    A candidate result is not automatically
    guaranteed to be the exact intended paper.

    The top result is the strongest candidate
    by evidence, not a canonical identity claim.
    """

    if not title.strip():

        return json_response({
            "status": "error",
            "error": (
                "title cannot be empty"
            )
        })

    try:

        results = find_research_papers(
            title=title,
            limit=limit
        )

        return json_response({
            "status": "success",
            "query": {
                "title": title
            },
            "count": len(results),
            "selection_note": (
                "Strongest candidate by title similarity;"
                " verify authors, year, DOI, venue, and"
                " source provenance before treating it as"
                " canonical."
            ),
            "candidates": results
        })

    except Exception as error:

        return json_response({
            "status": "error",
            "query": {
                "title": title
            },
            "error": str(error)
        })


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )