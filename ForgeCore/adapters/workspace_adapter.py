from typing import Any, Callable


class WorkspaceAdapter:
    def __init__(
        self,
        initialize_fn: Callable[..., dict[str, Any]],
        write_fn: Callable[..., dict[str, Any]],
        verify_fn: Callable[..., dict[str, Any]],
    ):
        self.initialize_fn = initialize_fn
        self.write_fn = write_fn
        self.verify_fn = verify_fn

    def initialize(
        self,
        project_name: str,
    ) -> dict[str, Any]:
        return self.initialize_fn(
            project_name=project_name
        )

    def write(
        self,
        project_name: str,
        relative_path: str,
        content: str,
    ) -> dict[str, Any]:
        return self.write_fn(
            project_name=project_name,
            relative_path=relative_path,
            content=content,
            overwrite=True,
        )

    def verify_syntax(
        self,
        project_name: str,
    ) -> dict[str, Any]:
        return self.verify_fn(
            project_name=project_name,
            check_type="syntax",
            timeout=30,
        )

    def verify_tests(
        self,
        project_name: str,
    ) -> dict[str, Any]:
        return self.verify_fn(
            project_name=project_name,
            check_type="pytest",
            timeout=60,
        )
