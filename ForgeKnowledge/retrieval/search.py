from retrieval.loader import (
    load_documents
)
from retrieval.normalizer import (
    normalize_text,
    tokenize
)


def build_searchable_text(
    document: dict
) -> str:

    parameter_text = " ".join(
        f"{name} {description}"
        for name, description
        in document.get(
            "parameters",
            {}
        ).items()
    )

    return " ".join([
        str(document.get(
            "symbol",
            ""
        ) or ""),

        str(document.get(
            "title",
            ""
        ) or ""),

        str(document.get(
            "signature",
            ""
        ) or ""),

        str(document.get(
            "description",
            ""
        ) or ""),

        parameter_text
    ])


def score_document(
    query: str,
    document: dict
) -> float:

    query_normalized = normalize_text(
        query
    )

    query_tokens = set(
        tokenize(query)
    )

    symbol = normalize_text(
        str(document.get(
            "symbol",
            ""
        ) or "")
    )

    title = normalize_text(
        str(document.get(
            "title",
            ""
        ) or "")
    )

    searchable_text = normalize_text(
        build_searchable_text(
            document
        )
    )

    searchable_tokens = set(
        tokenize(
            searchable_text
        )
    )

    score = 0.0

    if query_normalized == symbol:
        score += 100

    if (
        symbol
        and symbol in query_normalized
    ):
        score += 50

    if query_normalized == title:
        score += 40

    if (
        title
        and title in query_normalized
    ):
        score += 20

    if (
        query_normalized
        and query_normalized
        in searchable_text
    ):
        score += 15

    overlap = (
        query_tokens
        & searchable_tokens
    )

    score += len(overlap) * 3

    parameter_names = {
        normalize_text(name)
        for name in document.get(
            "parameters",
            {}
        )
    }

    parameter_overlap = (
        query_tokens
        & parameter_names
    )

    score += (
        len(parameter_overlap)
        * 8
    )

    return score


def search_documents(
    query: str,
    limit: int = 5
) -> list[dict]:

    if not query.strip():
        return []

    documents = load_documents()

    scored = []

    for document in documents:

        score = score_document(
            query,
            document
        )

        if score <= 0:
            continue

        scored.append({
            **document,
            "score": round(
                score,
                3
            )
        })

    scored.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )

    return scored[:limit]


def find_exact_symbol(
    symbol: str
) -> dict | None:

    documents = load_documents()

    target = normalize_text(
        symbol
    )

    for document in documents:

        current_symbol = normalize_text(
            str(document.get(
                "symbol",
                ""
            ) or "")
        )

        if current_symbol == target:
            return document

    return None
