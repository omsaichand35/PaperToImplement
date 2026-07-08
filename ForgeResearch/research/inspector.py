from research.fetcher import (
    fetch_page
)

from research.parser import (
    parse_page
)

from research.links import (
    extract_links,
    classify_links
)


def inspect_research_page(
    url: str
) -> dict:

    fetch_result = fetch_page(
        url
    )

    metadata = parse_page(
        fetch_result.html
    )

    links = extract_links(
        html=fetch_result.html,
        base_url=fetch_result.final_url
    )

    classified_links = classify_links(
        links
    )

    interesting_links = [
        link
        for link in classified_links
        if link["type"] != "other"
    ]

    return {
        "status": "success",

        "page": {
            "requested_url":
                fetch_result.requested_url,

            "final_url":
                fetch_result.final_url,

            "status_code":
                fetch_result.status_code,

            "content_type":
                fetch_result.content_type,

            "title":
                metadata.get("title"),

            "description":
                metadata.get("description")
        },

        "links": interesting_links,

        "stats": {
            "total_links":
                len(classified_links),

            "interesting_links":
                len(interesting_links)
        }
    }