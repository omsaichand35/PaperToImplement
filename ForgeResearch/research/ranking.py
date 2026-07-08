import re
from datetime import datetime
from urllib.parse import urlparse


def normalize_title(
    text: str
) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def extract_year(
    text: str
) -> int | None:

    match = re.search(
        r"(19|20)\d{2}",
        text
    )

    if not match:
        return None

    return int(
        match.group(0)
    )


def _candidate_urls(
    candidate: dict
) -> dict:

    urls = candidate.get(
        "urls"
    )

    if isinstance(urls, dict):
        return urls

    return {
        "research_landing_page": candidate.get(
            "landing_page_url"
        ),
        "direct_pdf": candidate.get(
            "pdf_url"
        ),
        "provider_record": candidate.get(
            "id"
        )
    }


def _candidate_landing_page_domain(
    candidate: dict
) -> str:

    urls = _candidate_urls(
        candidate
    )

    landing_page_url = urls.get(
        "research_landing_page"
    )

    if not landing_page_url:
        return ""

    parsed = urlparse(
        landing_page_url
    )

    return parsed.netloc.lower()


def _title_similarity_weight(
    query_title: str,
    candidate_title: str
) -> float:

    query = normalize_title(
        query_title
    )

    candidate = normalize_title(
        candidate_title
    )

    if not query or not candidate:
        return 0.0

    if query == candidate:
        return 50.0

    query_tokens = set(
        query.split()
    )

    candidate_tokens = set(
        candidate.split()
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        & candidate_tokens
    )

    union = (
        query_tokens
        | candidate_tokens
    )

    if not union:
        return 0.0

    return (
        len(overlap)
        / len(union)
        * 50.0
    )


def _author_evidence_weight(
    candidate: dict
) -> float:

    authors = candidate.get(
        "authors"
    ) or []

    if not authors:
        return 0.0

    if len(authors) == 1:
        return 10.0

    if len(authors) == 2:
        return 15.0

    return 20.0


def _year_consistency_weight(
    candidate: dict,
    baseline_year: int | None
) -> float:

    candidate_year = candidate.get(
        "publication_year"
    )

    if not isinstance(
        candidate_year,
        int
    ):
        return 0.0

    if baseline_year is not None:
        difference = abs(
            candidate_year - baseline_year
        )

        if difference == 0:
            return 10.0

        if difference == 1:
            return 6.0

        if difference <= 3:
            return 2.0

        return 0.0

    current_year = datetime.now().year

    if 1900 <= candidate_year <= current_year + 1:
        return 5.0

    return 0.0


def _identifier_quality_weight(
    candidate: dict
) -> float:

    doi = candidate.get(
        "doi"
    )

    if doi:
        if str(doi).lower().startswith(
            "https://doi.org/10.48550/arxiv."
        ):
            return 10.0

        return 8.0

    urls = _candidate_urls(
        candidate
    )

    if urls.get(
        "provider_record"
    ):
        return 3.0

    return 0.0


def _source_quality_weight(
    candidate: dict
) -> float:

    domain = _candidate_landing_page_domain(
        candidate
    )

    if not domain:
        return 0.0

    scholarly_hosts = {
        "arxiv.org",
        "www.arxiv.org",
        "doi.org",
        "www.doi.org",
        "nature.com",
        "www.nature.com",
        "science.org",
        "www.science.org",
        "acm.org",
        "www.acm.org",
        "ieee.org",
        "www.ieee.org",
        "proceedings.mlr.press",
        "aclanthology.org",
        "openaccess.thecvf.com",
        "pmc.ncbi.nlm.nih.gov"
    }

    if domain in scholarly_hosts:
        return 10.0

    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return 8.0

    if domain.endswith(".org"):
        return 6.0

    return 2.0


def _baseline_year_for_query(
    query_title: str,
    candidates: list[dict]
) -> int | None:

    query_year = extract_year(
        query_title
    )

    if query_year is not None:
        return query_year

    normalized_query_title = normalize_title(
        query_title
    )

    matching_years = []

    for candidate in candidates:

        candidate_title = candidate.get(
            "title"
        ) or ""

        if normalize_title(
            candidate_title
        ) != normalized_query_title:
            continue

        publication_year = candidate.get(
            "publication_year"
        )

        if isinstance(
            publication_year,
            int
        ):
            matching_years.append(
                publication_year
            )

    if not matching_years:
        return None

    return min(
        matching_years
    )


def title_similarity(
    query_title: str,
    candidate_title: str
) -> float:

    return _title_similarity_weight(
        query_title,
        candidate_title
    )


def rank_paper_candidates(
    query_title: str,
    candidates: list[dict]
) -> list[dict]:

    baseline_year = _baseline_year_for_query(
        query_title,
        candidates
    )

    ranked = []

    for candidate in candidates:

        candidate_title = (
            candidate.get("title")
            or ""
        )

        breakdown = {
            "title_similarity": round(
                title_similarity(
                    query_title,
                    candidate_title
                ),
                2
            ),
            "author_evidence": round(
                _author_evidence_weight(
                    candidate
                ),
                2
            ),
            "year_consistency": round(
                _year_consistency_weight(
                    candidate,
                    baseline_year
                ),
                2
            ),
            "identifier_quality": round(
                _identifier_quality_weight(
                    candidate
                ),
                2
            ),
            "source_quality": round(
                _source_quality_weight(
                    candidate
                ),
                2
            )
        }

        score = round(
            sum(
                breakdown.values()
            ),
            2
        )

        ranked.append({
            **candidate,
            "match_score": round(
                score,
                2
            ),
            "score_breakdown": breakdown,
            "ranking_note": (
                "strongest candidate by title similarity"
                if breakdown["title_similarity"] == 50.0
                else "ranked by combined evidence"
            )
        })

    ranked.sort(
        key=lambda item:
            item["match_score"],
        reverse=True
    )

    return ranked