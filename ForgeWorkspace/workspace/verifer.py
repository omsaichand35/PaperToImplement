import subprocess
import sys

from .paths import safe_project_path


ALLOWED_CHECKS = {
    "syntax",
    "pytest"
}


def verify_project(
    project_name: str,
    check_type: str = "syntax",
    timeout: int = 30
) -> dict:
    """
    Run an approved verification check
    inside a ForgeWorkspace project.

    Supported checks:
    - syntax
    - pytest
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

    check_type = (
        check_type
        .strip()
        .lower()
    )

    if check_type not in ALLOWED_CHECKS:
        return {
            "status": "error",
            "error": "Unsupported verification check",
            "check_type": check_type,
            "allowed_checks": sorted(
                ALLOWED_CHECKS
            )
        }

    command = build_command(
        check_type
    )

    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False
        )

    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "project_name": project_name,
            "check_type": check_type,
            "timeout_seconds": timeout,
            "stdout": error.stdout or "",
            "stderr": error.stderr or ""
        }

    return {
        "status": (
            "passed"
            if result.returncode == 0
            else "failed"
        ),
        "project_name": project_name,
        "check_type": check_type,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def get_test_executable() -> str:
    from pathlib import Path
    py311 = Path(r"C:\Users\omsai\AppData\Local\Programs\Python\Python311\python.exe")
    if py311.exists():
        return str(py311)
    venv_py = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def build_command(
    check_type: str
) -> list[str]:
    """
    Build only approved commands.

    No arbitrary user-provided shell command
    is ever executed.
    """
    python_exe = get_test_executable()

    if check_type == "syntax":
        return [
            python_exe,
            "-m",
            "compileall",
            "-q",
            "."
        ]

    if check_type == "pytest":
        return [
            python_exe,
            "-m",
            "pytest",
            "-q"
        ]

    raise ValueError(
        f"Unsupported check: {check_type}"
    )