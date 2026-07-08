import re


def normalize_text(
    text: str
) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9_.]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(
    text: str
) -> list[str]:

    normalized = normalize_text(text)

    return [
        token
        for token in normalized.split()
        if len(token) > 1
    ]
