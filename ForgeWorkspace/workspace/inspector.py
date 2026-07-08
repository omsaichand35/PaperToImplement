from pathlib import Path

from .paths import (
    safe_project_path
)


def build_tree(
    path: Path,
    prefix: str = ""
) -> list[str]:

    lines = []

    children = sorted(
        path.iterdir(),
        key=lambda item: (
            item.is_file(),
            item.name.lower()
        )
    )

    for index, child in enumerate(
        children
    ):

        is_last = (
            index
            == len(children) - 1
        )

        connector = (
            "└── "
            if is_last
            else "├── "
        )

        lines.append(
            f"{prefix}"
            f"{connector}"
            f"{child.name}"
        )

        if child.is_dir():

            extension = (
                "    "
                if is_last
                else "│   "
            )

            lines.extend(
                build_tree(
                    child,
                    prefix + extension
                )
            )

    return lines


def inspect_project(
    project_name: str
) -> dict:

    project_root = safe_project_path(
        project_name
    )

    if not project_root.exists():

        return {
            "status": "error",
            "error": "Project does not exist"
        }

    tree_lines = [
        project_root.name
    ]

    tree_lines.extend(
        build_tree(
            project_root
        )
    )

    return {
        "status": "success",
        "project_name": project_name,
        "tree": "\n".join(
            tree_lines
        )
    }

def list_project_files(
    project_name: str,
    extensions: list[str] | None = None
) -> dict:
    """
    Return structured file paths for an
    existing ForgeWorkspace project.

    Optionally filter by file extensions.
    """

    project_root = safe_project_path(
        project_name
    )

    if not project_root.exists():
        return {
            "status": "error",
            "error": "Project does not exist",
            "project_name": project_name
        }

    if not project_root.is_dir():
        return {
            "status": "error",
            "error": "Project path is not a directory",
            "project_name": project_name
        }

    normalized_extensions = None

    if extensions:
        normalized_extensions = {
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in extensions
        }

    files = []

    for path in project_root.rglob("*"):

        if not path.is_file():
            continue

        if (
            normalized_extensions
            and path.suffix.lower()
            not in normalized_extensions
        ):
            continue

        relative_path = path.relative_to(
            project_root
        )

        files.append(
            relative_path.as_posix()
        )

    files.sort()

    return {
        "status": "success",
        "project_name": project_name,
        "filters": {
            "extensions": (
                sorted(normalized_extensions)
                if normalized_extensions
                else None
            )
        },
        "count": len(files),
        "files": files
    }