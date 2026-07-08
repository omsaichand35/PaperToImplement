import json

from ForgeResearch.research.discovery import (
    find_research_papers
)


results = find_research_papers(
    title="Attention Is All You Need",
    limit=5
)


print(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    )
)