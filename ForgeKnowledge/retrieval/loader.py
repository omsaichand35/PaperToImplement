import json
from pathlib import Path


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

KNOWLEDGE_PATH = (
    BASE_DIR
    / "knowledge"
    / "pytorch_docs.json"
)


def load_documents() -> list[dict]:

    if not KNOWLEDGE_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge base not found: "
            f"{KNOWLEDGE_PATH}"
        )

    with open(
        KNOWLEDGE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    if not isinstance(
        documents,
        list
    ):

        raise ValueError(
            "Knowledge base must contain "
            "a JSON list"
        )

    return documents
