import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    BASE_DIR
    / "knowledge"
    / "pytorch_docs.json"
)


PYTORCH_APIS = {
    "torch.nn.Conv1d": (
        "https://docs.pytorch.org/docs/stable/"
        "generated/torch.nn.Conv1d.html"
    ),

    "torch.nn.ConvTranspose1d": (
        "https://docs.pytorch.org/docs/stable/"
        "generated/torch.nn.ConvTranspose1d.html"
    ),

    "torch.nn.Linear": (
        "https://docs.pytorch.org/docs/stable/"
        "generated/torch.nn.Linear.html"
    ),

    "torch.nn.LSTM": (
        "https://docs.pytorch.org/docs/stable/"
        "generated/torch.nn.LSTM.html"
    ),

    "torch.nn.BatchNorm1d": (
        "https://docs.pytorch.org/docs/stable/"
        "generated/torch.nn.BatchNorm1d.html"
    )
}


HEADERS = {
    "User-Agent": (
        "PaperForge-ForgeKnowledge/0.1 "
        "Documentation Research Client"
    )
}


def clean_text(text: str) -> str:
    """
    Normalize whitespace in extracted HTML text.
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def normalize_signature_text(text: str) -> str:
    """
    Collapse Sphinx-added spacing around punctuation in signatures.
    """

    normalized = clean_text(text)
    normalized = re.sub(r"\s+\.", ".", normalized)
    normalized = re.sub(r"\.\s+", ".", normalized)
    normalized = re.sub(r"\s+\(", "(", normalized)
    normalized = re.sub(r"\(\s+", "(", normalized)
    normalized = re.sub(r"\s+\)", ")", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r",\s+", ", ", normalized)
    normalized = re.sub(r"\s+=\s+", "=", normalized)

    return normalized


def extract_article_body(soup: BeautifulSoup) -> str | None:
    """
    Pull the structured article body from the JSON-LD metadata.
    """

    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):

        raw_text = script.string or script.get_text(strip=True)

        if not raw_text:
            continue

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            continue

        candidates = payload if isinstance(payload, list) else [payload]

        for item in candidates:

            if not isinstance(item, dict):
                continue

            if item.get("@type") != "Article":
                continue

            article_body = item.get("articleBody")

            if isinstance(article_body, str) and article_body:
                return clean_text(article_body)

    return None


def extract_first_sentence(text: str) -> str:
    """
    Keep the leading sentence so the ingested summary stays compact.
    """

    match = re.match(
        r"^(.+?[.!?])(?:\s|$)",
        text.strip()
    )

    if match:
        return clean_text(match.group(1))

    return clean_text(text)


def remove_parameter_name(
    description: str,
    name: str
) -> str:

    text = description.strip()

    if text.startswith(name):
        text = text[len(name):]

    text = re.sub(
        r"^\s*\([^)]*\)\s*[–—:-]\s*",
        "",
        text
    )

    text = text.lstrip(
        " –—:-"
    )

    return text.strip()


def extract_parameters_from_article_body(article_body: str) -> dict[str, str]:
    """
    Parse the parameter bullets from the JSON-LD article body.
    """

    parameters: dict[str, str] = {}

    match = re.search(
        r"Parameters:\s*(?P<body>.*?)(?:\s+Shape:|\s+Variables:|\s+Examples:|$)",
        article_body,
        re.IGNORECASE
    )

    if match is None:
        return parameters

    parameters_block = match.group("body")

    for item_match in re.finditer(
        r"\*\s+\*\*(?P<name>[^*]+)\*\*.*?[–-]\s*(?P<description>.*?)(?=(?:\s+\*\s+\*\*|$))",
        parameters_block,
        re.DOTALL
    ):

        name = clean_text(item_match.group("name"))
        description = clean_text(item_match.group("description"))

        if name and description:
            parameters[name] = description

    return parameters


def fetch_page(url: str) -> tuple[str, str]:
    """
    Download one documentation page and
    return both HTML and final resolved URL.
    """

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
        allow_redirects=True
    )

    response.raise_for_status()

    final_url = response.url
    html = response.text

    redirect_match = re.search(
        r'<link\s+rel="canonical"\s+href="(?P<href>[^"]+)"',
        html,
        re.IGNORECASE
    )

    if (
        len(html) < 5000
        and "Redirecting" in html
        and redirect_match is not None
    ):

        redirected_url = urljoin(
            final_url,
            redirect_match.group("href")
        )

        redirected_response = requests.get(
            redirected_url,
            headers=HEADERS,
            timeout=30,
            allow_redirects=True
        )

        redirected_response.raise_for_status()

        final_url = redirected_response.url
        html = redirected_response.text

    print(f"Requested URL: {url}")
    print(f"Final URL:     {final_url}")
    print(f"Status:        {response.status_code}")
    print(f"HTML size:     {len(html)}")

    return html, final_url


def find_api_definition(
    soup: BeautifulSoup,
    symbol: str
):
    """
    Find the definition element corresponding
    to the requested PyTorch symbol.
    """

    short_name = symbol.split(".")[-1]

    # First try exact id
    definition = soup.find(
        "dt",
        id=symbol
    )

    if definition is not None:
        return definition

    # Fallback:
    # inspect dt elements for symbol/name
    for dt in soup.find_all("dt"):

        text = clean_text(
            dt.get_text(
                " ",
                strip=True
            )
        )

        if (
            symbol in text
            or short_name in text
        ):
            return dt

    return None


def extract_definition_block(
    soup: BeautifulSoup,
    symbol: str
):
    """
    Return the API definition term and its detail block.
    """

    definition = find_api_definition(
        soup,
        symbol
    )

    if definition is None:
        return None, None

    details = definition.find_next_sibling("dd")

    return definition, details


def extract_signature(
    soup: BeautifulSoup,
    symbol: str
) -> str | None:

    definition, _ = extract_definition_block(
        soup,
        symbol
    )

    if definition is None:
        return None

    definition_text = clean_text(
        definition.get_text(
            " ",
            strip=True
        )
    )

    normalized_text = normalize_signature_text(
        definition_text
    )

    signature_match = re.search(
        rf"(class\s+{re.escape(symbol)}\s*\(.*?\))(?=\s*\[source\]|\s*#|$)",
        normalized_text
    )

    if signature_match is None:
        return None

    return clean_text(
        signature_match.group(1)
    )

def extract_description(
    soup: BeautifulSoup,
    symbol: str
) -> str | None:

    _, details = extract_definition_block(
        soup,
        symbol
    )

    if details is None:
        return None

    first_paragraph = details.find("p")

    if first_paragraph is not None:
        description = clean_text(
            first_paragraph.get_text(
                " ",
                strip=True
            )
        )

        if description:
            return extract_first_sentence(description)

    details_text = clean_text(
        details.get_text(
            " ",
            strip=True
        )
    )

    description_match = re.search(
        r"^(?P<description>.*?)(?:\s+Parameters:|\s+Shape:|\s+Variables:|\s+Examples:|$)",
        details_text
    )

    if description_match is None:
        return None

    description = clean_text(
        description_match.group("description")
    )

    if not description:
        return None

    return extract_first_sentence(description)


def extract_parameters(
    soup: BeautifulSoup,
    symbol: str
) -> dict:

    definition = find_api_definition(
        soup,
        symbol
    )

    if definition is None:
        return {}

    description_block = (
        definition.find_next_sibling("dd")
    )

    if description_block is None:
        return {}

    parameters = {}

    parameter_labels = description_block.find_all(
        string=lambda text:
        text
        and clean_text(text).lower()
        in {
            "parameters",
            "parameters:"
        }
    )

    for label in parameter_labels:

        label_element = label.parent

        if label_element is None:
            continue

        container = label_element.parent

        if container is None:
            continue

        for item in container.find_all("li"):

            item_text = clean_text(
                item.get_text(
                    " ",
                    strip=True
                )
            )

            strong = item.find("strong")

            if strong is not None:
                name = clean_text(
                    strong.get_text(
                        " ",
                        strip=True
                    )
                )
            else:
                match = re.match(
                    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
                    r"(?:\s*\([^)]*\))?\s*[–-]\s*(?P<description>.+)",
                    item_text
                )

                if match is None:
                    continue

                name = clean_text(match.group("name"))

            description = remove_parameter_name(
                item_text,
                name
            )

            if name and description:
                parameters[name] = description

        if parameters:
            return parameters

        for paragraph in container.find_all("p"):

            paragraph_text = clean_text(
                paragraph.get_text(
                    " ",
                    strip=True
                )
            )

            strong = paragraph.find("strong")

            if strong is not None:
                name = clean_text(
                    strong.get_text(
                        " ",
                        strip=True
                    )
                )
            else:
                match = re.match(
                    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
                    r"(?:\s*\([^)]*\))?\s*[–-]\s*(?P<description>.+)",
                    paragraph_text
                )

                if match is None:
                    continue

                name = clean_text(match.group("name"))

            description = remove_parameter_name(
                paragraph_text,
                name
            )

            if name and description:
                parameters[name] = description

        if parameters:
            return parameters

    return parameters


def extract_version(
    soup: BeautifulSoup,
    final_url: str
) -> str | None:

    match = re.search(
        r"/docs/([\d.]+)/",
        final_url
    )

    if match:
        return match.group(1)

    title = soup.title

    if title is None:
        return None

    title_text = clean_text(
        title.get_text(
            " ",
            strip=True
        )
    )

    match = re.search(
        r"PyTorch\s+([\d.]+)",
        title_text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def ingest_api(
    symbol: str,
    url: str
) -> dict:

    print(f"\nIngesting: {symbol}")

    html, final_url = fetch_page(url)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    signature = extract_signature(
        soup,
        symbol
    )

    description = extract_description(
        soup,
        symbol
    )

    parameters = extract_parameters(
        soup,
        symbol
    )

    version = extract_version(
        soup,
        final_url
    )

    return {
        "id": symbol
        .lower()
        .replace(".", "_"),

        "symbol": symbol,

        "title": symbol.split(".")[-1],

        "signature": signature,

        "description": description,

        "parameters": parameters,

        "source": {
            "type": "official_pytorch_docs",
            "url": final_url,
            "version": version
        }
    }


def save_documents(
    documents: list[dict]
) -> None:
    """
    Save structured documentation records.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():

    documents = []

    for symbol, url in PYTORCH_APIS.items():

        try:
            document = ingest_api(
                symbol,
                url
            )

            documents.append(document)

            print(
                f"Success: {symbol}"
            )

        except Exception as error:

            print(
                f"Failed: {symbol}"
            )

            print(
                f"Reason: {error}"
            )

    save_documents(documents)

    print(
        f"\nSaved {len(documents)} documents"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
