from pathlib import Path


def extract_pdf_pages(
    pdf_path: str | Path
) -> list[dict]:

    try:
        import fitz
    except ImportError as error:
        raise ImportError(
            "PyMuPDF is required for PaperUnderstanding. "
            "Install it with `pip install pymupdf`."
        ) from error

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f'PDF file "{pdf_path}" does not exist'
        )

    document = fitz.open(pdf_path)
    pages = []

    try:
        for page_index in range(len(document)):
            page = document[page_index]
            text = page.get_text("text").strip()
            pages.append({
                "page": page_index + 1,
                "text": text
            })
    finally:
        document.close()

    return pages


def build_numbered_text(
    pages: list[dict]
) -> str:

    blocks = []

    for page in pages:
        blocks.append(
            (
                f"==== PAGE {page['page']} ====\n"
                f"{page['text']}"
            ).strip()
        )

    return "\n\n".join(blocks)


if __name__ == "__main__":

    pages = extract_pdf_pages(
        "papers/Omsaichand_boppudi.pdf"
    )

    print(
        f"Extracted pages: {len(pages)}"
    )

    for page in pages[:3]:

        print(
            f"\n--- PAGE {page['page']} ---"
        )

        print(
            page["text"][:1000]
        )
