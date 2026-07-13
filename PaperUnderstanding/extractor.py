from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
_HYPHENATED_LINEBREAK_RE = re.compile(
    r"(?<=[A-Za-z])-\n(?=[a-z])"
)

_CAPTION_RE = re.compile(
    r"^\s*"
    r"(Figure|Fig\.?|Table|Algorithm|Alg\.?)"
    r"\s+([A-Za-z]?\d+(?:\.\d+)?)"
    r"[\s.:,-]*"
    r"(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)

_SECTION_RE = re.compile(
    r"^\s*"
    r"("
    r"(?:\d+(?:\.\d+)*)"
    r"|"
    r"(?:[A-Z](?:\.\d+)*)"
    r")"
    r"\.?\s+"
    r"(.+?)"
    r"\s*$"
)

_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references|bibliography)\s*$",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text while preserving paragraph structure.

    Important:
    - joins common word hyphenation across line breaks
    - collapses repeated spaces
    - avoids flattening all paragraphs into one line
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Example:
    # "architec-\nture" -> "architecture"
    text = _HYPHENATED_LINEBREAK_RE.sub("", text)

    lines = []
    for line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        lines.append(line)

    text = "\n".join(lines)
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox_from_block(
    block: dict[str, Any]
) -> tuple[float, float, float, float]:
    bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)

    if len(bbox) != 4:
        return 0.0, 0.0, 0.0, 0.0

    return tuple(
        _safe_float(value)
        for value in bbox
    )


def _extract_text_from_dict_block(
    block: dict[str, Any]
) -> str:
    """
    Extract text from a PyMuPDF dict block.

    Structure:
        block
          -> lines
              -> spans
    """
    line_texts: list[str] = []

    for line in block.get("lines", []):
        span_texts: list[str] = []

        for span in line.get("spans", []):
            text = span.get("text", "")
            if text:
                span_texts.append(text)

        line_text = "".join(span_texts).strip()

        if line_text:
            line_texts.append(line_text)

    return normalize_text("\n".join(line_texts))


def _detect_caption(
    text: str
) -> dict[str, Any] | None:
    match = _CAPTION_RE.match(text)

    if not match:
        return None

    label = match.group(1)
    number = match.group(2)
    description = match.group(3).strip()

    normalized_label = label.lower()

    if normalized_label.startswith("fig"):
        caption_type = "figure"
    elif normalized_label.startswith("table"):
        caption_type = "table"
    elif normalized_label.startswith("alg"):
        caption_type = "algorithm"
    else:
        caption_type = "caption"

    return {
        "type": caption_type,
        "label": label,
        "number": number,
        "text": text,
        "description": description or None,
    }


def _detect_section_heading(
    text: str
) -> dict[str, Any] | None:
    """
    Conservative section-heading detection.

    Examples:
        3. Approach
        3.2. Architecture
        A.3. Training Details
    """
    if not text:
        return None

    # Headings are normally short. This avoids classifying paragraphs.
    compact = " ".join(text.split())

    if len(compact) > 160:
        return None

    if _REFERENCE_HEADING_RE.match(compact):
        return {
            "number": None,
            "title": compact,
            "text": compact,
        }

    match = _SECTION_RE.match(compact)

    if not match:
        return None

    number = match.group(1).strip()
    title = match.group(2).strip()

    # Avoid accidental matches against equations or prose.
    if not title:
        return None

    return {
        "number": number,
        "title": title,
        "text": compact,
    }


def _reading_order_key(
    block: dict[str, Any],
    page_width: float
) -> tuple[int, float, float]:
    """
    Approximate reading order for common research-paper layouts.

    Handles:
    - single-column pages
    - two-column papers

    For two-column layouts:
        left-column blocks first,
        then right-column blocks.

    Full-width blocks are placed using vertical position.
    """
    x0, y0, x1, _ = _bbox_from_block(block)

    width = max(0.0, x1 - x0)

    if page_width <= 0:
        return 0, y0, x0

    full_width_threshold = page_width * 0.72
    midpoint = page_width / 2.0

    # Full-width title/abstract/figure-caption blocks.
    if width >= full_width_threshold:
        return 0, y0, x0

    # Typical two-column ordering.
    if x0 < midpoint:
        column = 1
    else:
        column = 2

    return column, y0, x0


def _extract_page_blocks(
    page: Any
) -> list[dict[str, Any]]:
    """
    Extract text blocks with bounding boxes and semantic hints.
    """
    page_dict = page.get_text("dict")

    raw_blocks = page_dict.get("blocks", [])

    text_blocks: list[dict[str, Any]] = []

    for raw_index, block in enumerate(raw_blocks):
        # PyMuPDF block type:
        # 0 = text
        # 1 = image
        block_type = block.get("type")

        if block_type != 0:
            continue

        text = _extract_text_from_dict_block(block)

        if not text:
            continue

        bbox = _bbox_from_block(block)

        caption = _detect_caption(text)
        section = _detect_section_heading(text)

        text_blocks.append({
            "block_index": raw_index,
            "bbox": [
                round(bbox[0], 2),
                round(bbox[1], 2),
                round(bbox[2], 2),
                round(bbox[3], 2),
            ],
            "text": text,
            "caption": caption,
            "section": section,
        })

    page_width = _safe_float(page.rect.width)

    text_blocks.sort(
        key=lambda item: _reading_order_key(
            item,
            page_width
        )
    )

    # Re-index after reading-order sorting.
    for index, block in enumerate(text_blocks):
        block["reading_order"] = index

    return text_blocks


# ---------------------------------------------------------------------
# Visual asset extraction (Vision support)
# ---------------------------------------------------------------------

_ARCHITECTURE_KEYWORDS_RE = re.compile(
    r"\b(architecture|overview|framework|pipeline|model|network|diagram|flowchart|schema|structure)\b",
    flags=re.IGNORECASE,
)


def _is_architecture_or_figure_context(
    captions: list[dict[str, Any]],
    page_text: str,
) -> bool:
    if captions:
        return True
    return bool(_ARCHITECTURE_KEYWORDS_RE.search(page_text))


def extract_page_visual_assets(
    page: Any,
    page_number: int,
    captions: list[dict[str, Any]],
    extract_images: bool = True,
    render_architecture_pages: bool = True,
    max_images_per_page: int = 3,
    min_dimension: int = 60,
) -> list[dict[str, Any]]:
    """
    Extract visual assets (embedded figures or architecture page renders)
    from a PDF page for vision multimodal analysis.
    """
    visual_assets: list[dict[str, Any]] = []

    if not extract_images:
        return visual_assets

    # Match captions to extracted images sequentially when available.
    caption_texts = [
        capt.get("text", "")
        for capt in captions
        if capt.get("text")
    ]

    try:
        raw_images = page.get_images(full=True)
    except Exception:
        raw_images = []

    asset_index = 1
    for img_info in raw_images:
        if len(visual_assets) >= max_images_per_page:
            break

        xref = img_info[0]
        try:
            pix = page.parent.extract_image(xref)
            img_bytes = pix.get("image")
            width = int(pix.get("width", 0))
            height = int(pix.get("height", 0))

            if not img_bytes or width < min_dimension or height < min_dimension:
                continue

            caption_match = (
                caption_texts[asset_index - 1]
                if (asset_index - 1) < len(caption_texts)
                else (caption_texts[0] if caption_texts else None)
            )

            b64_str = base64.b64encode(img_bytes).decode("ascii")
            visual_assets.append({
                "image_id": f"p{page_number}_img{asset_index}",
                "page": page_number,
                "width": width,
                "height": height,
                "base64": b64_str,
                "caption": caption_match,
                "source_type": "embedded_image",
            })
            asset_index += 1
        except Exception:
            continue

    # If no embedded image objects were found on a page with architecture figures
    # (common for vector graphics diagrams in PDFs), optionally render the page.
    if not visual_assets and render_architecture_pages:
        page_text = page.get_text() if hasattr(page, "get_text") else ""
        if _is_architecture_or_figure_context(captions, page_text):
            try:
                pix = page.get_pixmap(dpi=120)
                width = int(pix.width)
                height = int(pix.height)
                img_bytes = pix.tobytes("png")
                b64_str = base64.b64encode(img_bytes).decode("ascii")
                caption_match = caption_texts[0] if caption_texts else "Page Architecture Diagram"
                visual_assets.append({
                    "image_id": f"p{page_number}_render1",
                    "page": page_number,
                    "width": width,
                    "height": height,
                    "base64": b64_str,
                    "caption": caption_match,
                    "source_type": "page_render",
                })
            except Exception:
                pass

    return visual_assets


# ---------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------

def extract_pdf_pages(
    pdf_path: str | Path,
    extract_images: bool = True,
    render_architecture_pages: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract structured page-level information from a research PDF.

    Returned page structure:

    {
        "page": 1,
        "width": ...,
        "height": ...,
        "text": "...",
        "blocks": [...],
        "captions": [...],
        "sections": [...],
        "image_count": 2,
        "images": [...]
    }

    Compatibility:
    Existing code using page["page"] and page["text"] still works.
    """
    try:
        # pyrefly: ignore [missing-import]
        import fitz
    except ImportError as error:
        raise ImportError(
            "PyMuPDF is required for PaperUnderstanding. "
            "Install it with: pip install pymupdf"
        ) from error

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f'PDF file "{pdf_path}" does not exist'
        )

    if not pdf_path.is_file():
        raise ValueError(
            f'PDF path "{pdf_path}" is not a file'
        )

    document = fitz.open(pdf_path)

    pages: list[dict[str, Any]] = []

    try:
        for page_index in range(len(document)):
            page = document[page_index]

            blocks = _extract_page_blocks(page)

            page_text = "\n\n".join(
                block["text"]
                for block in blocks
                if block["text"]
            ).strip()

            captions = [
                block["caption"]
                for block in blocks
                if block["caption"] is not None
            ]

            sections = [
                block["section"]
                for block in blocks
                if block["section"] is not None
            ]

            try:
                images = page.get_images(full=True)
                image_count = len(images)
            except Exception:
                image_count = 0

            visual_assets = extract_page_visual_assets(
                page=page,
                page_number=page_index + 1,
                captions=captions,
                extract_images=extract_images,
                render_architecture_pages=render_architecture_pages,
            )

            pages.append({
                "page": page_index + 1,
                "width": round(
                    _safe_float(page.rect.width),
                    2
                ),
                "height": round(
                    _safe_float(page.rect.height),
                    2
                ),
                "text": page_text,
                "blocks": blocks,
                "captions": captions,
                "sections": sections,
                "image_count": image_count,
                "images": visual_assets,
            })

    finally:
        document.close()

    return pages


# ---------------------------------------------------------------------
# Prompt serialization
# ---------------------------------------------------------------------

def _format_section_marker(
    section: dict[str, Any]
) -> str:
    number = section.get("number")
    title = section.get("title")

    if number:
        return f"{number}. {title}"

    return str(title or section.get("text") or "")


def _format_caption_marker(
    caption: dict[str, Any]
) -> str:
    caption_type = caption.get(
        "type",
        "caption"
    ).upper()

    number = caption.get("number")
    text = caption.get("text", "")

    if number:
        return (
            f"[{caption_type} {number}] "
            f"{text}"
        )

    return f"[{caption_type}] {text}"


def build_numbered_text(
    pages: list[dict[str, Any]]
) -> str:
    """
    Convert structured pages into LLM-friendly text.

    Preserves:
    - page boundaries
    - section hints
    - caption hints
    - image-presence hints

    This function keeps compatibility with client.py.
    """
    blocks: list[str] = []

    for page in pages:
        page_number = page["page"]

        header_lines = [
            f"==== PAGE {page_number} ===="
        ]

        sections = page.get("sections", [])
        captions = page.get("captions", [])
        image_count = page.get("image_count", 0)

        if sections:
            section_text = " | ".join(
                _format_section_marker(section)
                for section in sections
            )
            header_lines.append(
                f"[DETECTED SECTIONS] {section_text}"
            )

        if image_count:
            header_lines.append(
                f"[PAGE CONTAINS {image_count} EMBEDDED IMAGE(S)]"
            )

        images = page.get("images", [])
        if images:
            for img in images:
                caption_hint = img.get("caption") or "Architecture figure asset"
                header_lines.append(
                    f"[VISUAL ASSET {img.get('image_id')} ({img.get('width')}x{img.get('height')})]: {caption_hint}"
                )

        if captions:
            header_lines.append(
                "[DETECTED FIGURE/TABLE CAPTIONS]"
            )

            for caption in captions:
                header_lines.append(
                    _format_caption_marker(caption)
                )

        header = "\n".join(header_lines)

        page_text = page.get(
            "text",
            ""
        ).strip()

        blocks.append(
            f"{header}\n\n{page_text}".strip()
        )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------
# Better chunking
# ---------------------------------------------------------------------

def split_pages_into_chunks(
    pages: list[dict[str, Any]],
    pages_per_chunk: int = 4,
    overlap_pages: int = 0
) -> list[list[dict[str, Any]]]:
    """
    Split pages into overlapping chunks.

    Example:
        pages_per_chunk=4
        overlap_pages=1

        chunk 1: pages 1-4
        chunk 2: pages 4-7
        chunk 3: pages 7-10

    This reduces architecture loss across page boundaries.
    """
    if pages_per_chunk <= 0:
        raise ValueError(
            "pages_per_chunk must be greater than 0"
        )

    if overlap_pages < 0:
        raise ValueError(
            "overlap_pages cannot be negative"
        )

    if overlap_pages >= pages_per_chunk:
        raise ValueError(
            "overlap_pages must be smaller than "
            "pages_per_chunk"
        )

    if not pages:
        return []

    step = pages_per_chunk - overlap_pages

    chunks: list[list[dict[str, Any]]] = []

    start = 0

    while start < len(pages):
        end = min(
            start + pages_per_chunk,
            len(pages)
        )

        chunk = pages[start:end]

        if chunk:
            chunks.append(chunk)

        if end >= len(pages):
            break

        start += step

    return chunks


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def summarize_extraction(
    pages: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Produce extraction diagnostics useful for debugging.
    """
    total_blocks = sum(
        len(page.get("blocks", []))
        for page in pages
    )

    total_captions = sum(
        len(page.get("captions", []))
        for page in pages
    )

    total_sections = sum(
        len(page.get("sections", []))
        for page in pages
    )

    total_images = sum(
        int(page.get("image_count", 0))
        for page in pages
    )

    total_visual_assets = sum(
        len(page.get("images", []))
        for page in pages
    )

    empty_pages = [
        page["page"]
        for page in pages
        if not page.get("text", "").strip()
    ]

    return {
        "page_count": len(pages),
        "text_block_count": total_blocks,
        "caption_count": total_captions,
        "section_count": total_sections,
        "embedded_image_count": total_images,
        "visual_asset_count": total_visual_assets,
        "empty_pages": empty_pages,
    }


# ---------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Extract structured text and metadata "
            "from a research PDF."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file"
    )

    parser.add_argument(
        "--preview-pages",
        type=int,
        default=3,
        help="Number of extracted pages to preview"
    )

    args = parser.parse_args()

    extracted_pages = extract_pdf_pages(
        args.pdf_path
    )

    diagnostics = summarize_extraction(
        extracted_pages
    )

    print("\n=== EXTRACTION SUMMARY ===")
    print(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False
        )
    )

    preview_count = min(
        args.preview_pages,
        len(extracted_pages)
    )

    print("\n=== TEXT PREVIEW ===")

    for page in extracted_pages[:preview_count]:
        print(
            f"\n--- PAGE {page['page']} ---"
        )

        print(
            page["text"][:2000]
        )

        if page.get("sections"):
            print("\nDetected sections:")
            for section in page["sections"]:
                print(
                    "  -",
                    _format_section_marker(section)
                )

        if page.get("captions"):
            print("\nDetected captions:")
            for caption in page["captions"]:
                print(
                    "  -",
                    _format_caption_marker(caption)
                )

    print("\n=== CHUNK PREVIEW ===")

    chunks = split_pages_into_chunks(
        extracted_pages,
        pages_per_chunk=4,
        overlap_pages=1
    )

    for index, chunk in enumerate(
        chunks,
        start=1
    ):
        page_numbers = [
            page["page"]
            for page in chunk
        ]

        print(
            f"Chunk {index}: pages {page_numbers}"
        )