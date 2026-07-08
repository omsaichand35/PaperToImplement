from typing import Any, Callable


class ResearchAdapter:
    def __init__(
        self,
        find_papers_fn: Callable[..., list[dict[str, Any]]],
        inspect_page_fn: Callable[..., dict[str, Any]],
    ):
        self.find_papers_fn = find_papers_fn
        self.inspect_page_fn = inspect_page_fn

    def find_papers(
        self,
        title: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.find_papers_fn(
            title=title,
            limit=limit,
        )

    def inspect_page(
        self,
        url: str,
    ) -> dict[str, Any]:
        return self.inspect_page_fn(
            url=url,
        )
