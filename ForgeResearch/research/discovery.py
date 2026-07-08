from research.ranking import rank_paper_candidates

import requests


OPENALEX_WORKS_URL = (
    "https://api.openalex.org/works"
)


DEFAULT_HEADERS = {
    "User-Agent": (
        "PaperForge-ForgeResearch/0.1 "
        "(Research Discovery Client)"
    )
}


def find_research_papers(
    title: str,
    limit: int = 5
) -> list[dict]:
    """
    Search for research papers by title.

    Returns ranked candidate works from OpenAlex.
    """

    title = title.strip()

    if not title:
        raise ValueError(
            "title cannot be empty"
        )

    limit = max(
        1,
        min(limit, 10)
    )

    response = requests.get(
        OPENALEX_WORKS_URL,
        headers=DEFAULT_HEADERS,
        params={
            "search": title,
            "per-page": limit
        },
        timeout=30
    )

    response.raise_for_status()

    payload = response.json()

    results = []

    for work in payload.get(
        "results",
        []
    ):

        results.append(
            normalize_work(work)
        )

    return rank_paper_candidates(
        query_title=title,
        candidates=results
    )


def normalize_work(
    work: dict
) -> dict:
    """
    Convert OpenAlex work data into
    ForgeResearch's own stable structure.
    """

    authors = []

    for authorship in work.get(
        "authorships",
        []
    ):

        author = authorship.get(
            "author",
            {}
        )

        name = author.get(
            "display_name"
        )

        if name:
            authors.append(name)

    primary_location = (
        work.get("primary_location")
        or {}
    )

    landing_page_url = (
        primary_location.get(
            "landing_page_url"
        )
    )

    pdf_url = (
        primary_location.get(
            "pdf_url"
        )
    )

    open_access = (
        work.get("open_access")
        or {}
    )

    best_oa_location = (
        work.get("best_oa_location")
        or {}
    )

    # Useful fallback when primary location
    # does not expose a page/PDF.
    if not landing_page_url:
        landing_page_url = (
            best_oa_location.get(
                "landing_page_url"
            )
        )

    if not pdf_url:
        pdf_url = (
            best_oa_location.get(
                "pdf_url"
            )
        )

    return {
        "id": work.get("id"),

        "doi": work.get("doi"),

        "title": work.get(
            "display_name"
        ),

        "publication_year": work.get(
            "publication_year"
        ),

        "authors": authors,

        "type": work.get("type"),

        "cited_by_count": work.get(
            "cited_by_count"
        ),

        "urls": {
            "research_landing_page":
                landing_page_url,

            "direct_pdf":
                pdf_url,

            "provider_record":
                work.get("id")
        },

        "is_open_access":
            open_access.get(
                "is_oa"
            ),

        "open_access_status":
            open_access.get(
                "oa_status"
            ),

        "source": {
            "provider": "openalex"
        }
    }