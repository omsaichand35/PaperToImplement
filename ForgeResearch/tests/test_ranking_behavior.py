from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from research.discovery import normalize_work
from research.ranking import rank_paper_candidates


def test_ranker_prefers_older_exact_match_over_suspicious_duplicate():

    candidates = [
        {
            "title": "Attention Is All You Need",
            "publication_year": 2025,
            "authors": ["A. Example", "B. Example"],
            "doi": "https://doi.org/10.9999/example.2025.1",
            "urls": {
                "research_landing_page": "https://thirdparty.example/paper",
                "direct_pdf": "https://thirdparty.example/paper.pdf",
                "provider_record": "https://openalex.org/W1"
            }
        },
        {
            "title": "Attention Is All You Need",
            "publication_year": 2017,
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "doi": "https://doi.org/10.48550/arXiv.1706.03762",
            "urls": {
                "research_landing_page": "https://arxiv.org/abs/1706.03762",
                "direct_pdf": "https://arxiv.org/pdf/1706.03762.pdf",
                "provider_record": "https://openalex.org/W2"
            }
        }
    ]

    ranked = rank_paper_candidates(
        query_title="Attention Is All You Need",
        candidates=candidates
    )

    assert ranked[0]["publication_year"] == 2017
    assert ranked[0]["score_breakdown"]["source_quality"] > ranked[1]["score_breakdown"]["source_quality"]
    assert ranked[0]["match_score"] > ranked[1]["match_score"]


def test_normalize_work_exposes_explicit_url_semantics():

    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1234/example",
        "display_name": "Example Paper",
        "publication_year": 2024,
        "authorships": [
            {"author": {"display_name": "Ada Lovelace"}}
        ],
        "type": "article",
        "cited_by_count": 10,
        "primary_location": {
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf"
        },
        "open_access": {
            "is_oa": True,
            "oa_status": "gold"
        }
    }

    normalized = normalize_work(
        work
    )

    assert normalized["urls"]["research_landing_page"] == "https://example.org/paper"
    assert normalized["urls"]["direct_pdf"] == "https://example.org/paper.pdf"
    assert normalized["urls"]["provider_record"] == "https://openalex.org/W123"
    assert normalized["source"]["provider"] == "openalex"