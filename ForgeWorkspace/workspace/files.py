from workspace.paths import (
    safe_project_path,
    safe_file_path
)


def write_project_file(
    project_name: str,
    relative_path: str,
    content: str,
    overwrite: bool = False
) -> dict:
    """
    Write a text file safely inside
    an existing project workspace.
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

    file_path = safe_file_path(
        project_name=project_name,
        relative_path=relative_path
    )

    if (
        file_path.exists()
        and not overwrite
    ):
        return {
            "status": "error",
            "error": "File already exists",
            "project_name": project_name,
            "relative_path": relative_path,
            "hint": (
                "Set overwrite=true only if "
                "replacement is intentional"
            )
        }

    if (
        file_path.exists()
        and not file_path.is_file()
    ):
        return {
            "status": "error",
            "error": (
                "Target path exists but "
                "is not a file"
            )
        }

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path.write_text(
        content,
        encoding="utf-8"
    )

    return {
        "status": "success",
        "project_name": project_name,
        "relative_path": relative_path,
        "bytes_written": len(
            content.encode("utf-8")
        ),
        "overwritten": overwrite
    }

def read_project_file(
    project_name: str,
    relative_path: str
) -> dict:
    """
    Read a UTF-8 text file safely from
    an existing project workspace.
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

    file_path = safe_file_path(
        project_name=project_name,
        relative_path=relative_path
    )

    if not file_path.exists():
        return {
            "status": "error",
            "error": "File does not exist",
            "project_name": project_name,
            "relative_path": relative_path
        }

    if not file_path.is_file():
        return {
            "status": "error",
            "error": "Path is not a file",
            "project_name": project_name,
            "relative_path": relative_path
        }

    try:
        content = file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        return {
            "status": "error",
            "error": (
                "File is not valid UTF-8 text"
            ),
            "project_name": project_name,
            "relative_path": relative_path
        }

    return {
        "status": "success",
        "project_name": project_name,
        "relative_path": relative_path,
        "content": content,
        "characters": len(content),
        "bytes": len(
            content.encode("utf-8")
        )
    }