from dataclasses import dataclass

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "PaperForge-ForgeResearch/0.1 "
        "(Research Page Inspector)"
    ),
    "Accept": (
        "text/html,"
        "application/xhtml+xml"
    )
}


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    html: str


def fetch_page(
    url: str,
    timeout: int = 20
) -> FetchResult:
    """
    Fetch a public HTML page.

    Returns metadata plus the downloaded HTML.
    """

    if not url.strip():
        raise ValueError(
            "URL cannot be empty"
        )

    if not url.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "URL must start with "
            "http:// or https://"
        )

    response = requests.get(
        url,
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        allow_redirects=True
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("Content-Type", "")
        .lower()
    )

    if "text/html" not in content_type:
        raise ValueError(
            "Expected an HTML page, "
            f"received: {content_type}"
        )

    return FetchResult(
        requested_url=url,
        final_url=response.url,
        status_code=response.status_code,
        content_type=content_type,
        html=response.text
    )