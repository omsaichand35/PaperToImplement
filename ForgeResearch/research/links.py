from urllib.parse import (
    urljoin,
    urlparse
)

from bs4 import BeautifulSoup


def extract_links(
    html: str,
    base_url: str
) -> list[dict]:

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    results = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True
    ):

        href = anchor.get(
            "href",
            ""
        ).strip()

        if not href:
            continue

        absolute_url = urljoin(
            base_url,
            href
        )

        parsed = urlparse(
            absolute_url
        )

        if parsed.scheme not in {
            "http",
            "https"
        }:
            continue

        if absolute_url in seen:
            continue

        seen.add(
            absolute_url
        )

        text = " ".join(
            anchor.get_text(
                " ",
                strip=True
            ).split()
        )

        results.append({
            "text": text or None,
            "url": absolute_url,
            "domain": parsed.netloc.lower()
        })

    return results

def classify_link(
    link: dict
) -> str:

    url = (
        link.get("url")
        or ""
    ).lower()

    text = (
        link.get("text")
        or ""
    ).lower()

    domain = (
        link.get("domain")
        or ""
    ).lower()

    searchable = (
        f"{text} {url}"
    )

    # Repository candidates
    if domain in {
        "github.com",
        "www.github.com",
        "gitlab.com",
        "www.gitlab.com"
    }:
        return "repository"

    # PDF candidates
    if (
        url.endswith(".pdf")
        or "/pdf/" in url
    ):
        return "paper_pdf"

    # Supplementary material
    supplementary_terms = {
        "supplementary",
        "supplemental",
        "appendix",
        "additional material"
    }

    if any(
        term in searchable
        for term in supplementary_terms
    ):
        return "supplementary"

    # Dataset candidates
    dataset_terms = {
        "dataset",
        "data set",
        "download data"
    }

    if any(
        term in searchable
        for term in dataset_terms
    ):
        return "dataset"

    # Code candidates
    code_terms = {
        "code",
        "implementation",
        "source code"
    }

    if any(
        term in searchable
        for term in code_terms
    ):
        return "code_candidate"

    return "other"


def classify_links(
    links: list[dict]
) -> list[dict]:

    return [
        {
            **link,
            "type": classify_link(link)
        }
        for link in links
    ]