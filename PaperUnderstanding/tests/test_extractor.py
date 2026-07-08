import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from extractor import build_numbered_text  # noqa: E402


def test_build_numbered_text_includes_all_pages():

    pages = [
        {
            "page": 1,
            "text": "First page text"
        },
        {
            "page": 2,
            "text": "Second page text"
        },
        {
            "page": 3,
            "text": "Third page text"
        }
    ]

    result = build_numbered_text(
        pages
    )

    assert "==== PAGE 1 ====" in result
    assert "==== PAGE 2 ====" in result
    assert "==== PAGE 3 ====" in result
    assert "First page text" in result
    assert "Second page text" in result
    assert "Third page text" in result
