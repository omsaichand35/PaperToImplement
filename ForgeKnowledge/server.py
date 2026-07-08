import json

from mcp.server.fastmcp import FastMCP

from retrieval.loader import (
    load_documents
)
from retrieval.search import (
    find_exact_symbol,
    search_documents
)


mcp = FastMCP(
    "ForgeKnowledge"
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
def search_docs(
    query: str,
    limit: int = 5
) -> str:
    """
    Search trusted indexed PyTorch documentation.

    Use when:
    - the exact API symbol is unknown
    - the question is conceptual
    - multiple APIs may be relevant
    """

    if not query.strip():

        return json_response({
            "status": "error",
            "error": (
                "query cannot be empty"
            )
        })

    limit = max(
        1,
        min(limit, 10)
    )

    results = search_documents(
        query=query,
        limit=limit
    )

    return json_response({
        "status": "success",
        "query": query,
        "count": len(results),
        "results": results
    })


@mcp.tool()
def get_api_reference(
    symbol: str
) -> str:
    """
    Retrieve structured documentation for
    an exact PyTorch API symbol.

    Example:
    torch.nn.Conv1d
    """

    if not symbol.strip():

        return json_response({
            "status": "error",
            "error": (
                "symbol cannot be empty"
            )
        })

    result = find_exact_symbol(
        symbol
    )

    if result is None:

        return json_response({
            "status": "not_found",
            "symbol": symbol,
            "result": None
        })

    return json_response({
        "status": "success",
        "symbol": symbol,
        "result": result
    })


@mcp.tool()
def get_parameter_reference(
    symbol: str,
    parameter: str
) -> str:
    """
    Retrieve documentation for one parameter
    of an exact PyTorch API symbol.

    Example:
    symbol = torch.nn.ConvTranspose1d
    parameter = output_padding
    """

    if not symbol.strip() or not parameter.strip():

        return json_response({
            "status": "error",
            "error": (
                "symbol and parameter cannot be empty"
            )
        })

    document = find_exact_symbol(
        symbol
    )

    if document is None:

        return json_response({
            "status": "not_found",
            "reason": "symbol_not_found",
            "symbol": symbol
        })

    parameters = document.get(
        "parameters",
        {}
    )

    target = parameter.lower().strip()

    for name, description in (
        parameters.items()
    ):

        if name.lower() == target:

            return json_response({
                "status": "success",
                "symbol": symbol,
                "parameter": name,
                "description": (
                    description
                ),
                "source": document.get(
                    "source"
                )
            })

    return json_response({
        "status": "not_found",
        "reason": "parameter_not_found",
        "symbol": symbol,
        "parameter": parameter
    })


@mcp.tool()
def list_indexed_apis() -> str:
    """
    List all PyTorch API symbols currently
    indexed by ForgeKnowledge.
    """

    documents = load_documents()

    symbols = [
        document.get("symbol")
        for document in documents
        if document.get("symbol")
    ]

    return json_response({
        "status": "success",
        "count": len(symbols),
        "symbols": symbols
    })


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )
