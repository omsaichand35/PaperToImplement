from bs4 import BeautifulSoup


def clean_text(
    text: str | None
) -> str | None:

    if not text:
        return None

    return " ".join(
        text.split()
    )


def extract_title(
    soup: BeautifulSoup
) -> str | None:

    # Prefer OpenGraph metadata
    og_title = soup.find(
        "meta",
        property="og:title"
    )

    if (
        og_title
        and og_title.get("content")
    ):
        return clean_text(
            og_title["content"]
        )

    # Standard HTML title
    if soup.title:
        return clean_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )

    # Fallback to first h1
    h1 = soup.find("h1")

    if h1:
        return clean_text(
            h1.get_text(
                " ",
                strip=True
            )
        )

    return None


def extract_description(
    soup: BeautifulSoup
) -> str | None:

    og_description = soup.find(
        "meta",
        property="og:description"
    )

    if (
        og_description
        and og_description.get("content")
    ):
        return clean_text(
            og_description["content"]
        )

    meta_description = soup.find(
        "meta",
        attrs={"name": "description"}
    )

    if (
        meta_description
        and meta_description.get("content")
    ):
        return clean_text(
            meta_description["content"]
        )

    return None


def parse_page(
    html: str
) -> dict:

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    return {
        "title": extract_title(soup),
        "description": extract_description(soup)
    }