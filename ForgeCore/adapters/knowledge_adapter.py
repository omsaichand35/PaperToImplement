from typing import Any, Callable


class KnowledgeAdapter:
    def __init__(
        self,
        search_fn: Callable[..., list[dict[str, Any]]],
        exact_symbol_fn: Callable[..., dict[str, Any] | None],
    ):
        self.search_fn = search_fn
        self.exact_symbol_fn = exact_symbol_fn

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.search_fn(
            query=query,
            limit=limit,
        )

    def get_symbol(
        self,
        symbol: str,
    ) -> dict[str, Any] | None:
        return self.exact_symbol_fn(
            symbol=symbol,
        )
