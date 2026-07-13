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


def test_build_numbered_text_includes_visual_assets():
    pages = [
        {
            "page": 1,
            "text": "Page with figure",
            "images": [
                {
                    "image_id": "p1_img1",
                    "width": 640,
                    "height": 480,
                    "caption": "Figure 1: Architecture of PointDiT",
                }
            ],
        }
    ]

    result = build_numbered_text(pages)
    assert "[VISUAL ASSET p1_img1 (640x480)]: Figure 1: Architecture of PointDiT" in result


def test_summarize_extraction_counts_visual_assets():
    from extractor import summarize_extraction

    pages = [
        {
            "page": 1,
            "text": "Intro",
            "image_count": 2,
            "images": [{"image_id": "p1_img1"}],
        },
        {
            "page": 2,
            "text": "Method",
            "image_count": 0,
            "images": [{"image_id": "p2_render1"}, {"image_id": "p2_render2"}],
        },
    ]

    summary = summarize_extraction(pages)
    assert summary["visual_asset_count"] == 3
    assert summary["embedded_image_count"] == 2
