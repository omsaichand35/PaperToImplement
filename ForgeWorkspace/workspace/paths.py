from pathlib import Path


WORKSPACE_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    / "projects"
)


def ensure_workspace_root() -> Path:
    """
    Ensure that the workspace root exists.
    """

    WORKSPACE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    return WORKSPACE_ROOT


def safe_project_path(
    project_name: str
) -> Path:
    """
    Resolve a project path safely inside
    the ForgeWorkspace projects directory.
    """

    project_name = project_name.strip()

    if not project_name:
        raise ValueError(
            "project_name cannot be empty"
        )

    root = ensure_workspace_root().resolve()

    project_path = (
        root
        / project_name
    ).resolve()

    try:
        project_path.relative_to(root)

    except ValueError:
        raise ValueError(
            "Project path escapes workspace root"
        )

    return project_path

def safe_file_path(
    project_name: str,
    relative_path: str
) -> Path:
    """
    Resolve a file path safely inside
    a specific ForgeWorkspace project.
    """

    project_root = safe_project_path(
        project_name
    ).resolve()

    relative_path = relative_path.strip()

    if not relative_path:
        raise ValueError(
            "relative_path cannot be empty"
        )

    file_path = (
        project_root
        / relative_path
    ).resolve()

    try:
        file_path.relative_to(
            project_root
        )

    except ValueError:
        raise ValueError(
            "File path escapes project root"
        )

    return file_path