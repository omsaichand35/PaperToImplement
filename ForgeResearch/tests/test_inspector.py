import json

from ForgeResearch.research.inspector import (
    inspect_research_page
)


result = inspect_research_page(
    "https://arxiv.org/abs/1706.03762"
)


print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )
)