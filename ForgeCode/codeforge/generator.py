import json
import os

from openai import AsyncOpenAI
from pydantic import ValidationError

from paperforge_env import load_project_env

load_project_env()

from .prompts import FILE_GENERATOR_SYSTEM_PROMPT

from .schemas import (
    ImplementationPlan,
    GeneratedFile,
    extract_json,
    stream_chat_with_retries,
)


client = AsyncOpenAI(
    base_url=(
        "https://integrate.api.nvidia.com/v1"
    ),
    max_retries=3,
    timeout=300.0,
)


async def generate_file(
    spec: str,
    plan: dict,
    target_path: str,
    dependency_context: dict[str, str] | None = None,
    error_context: str | None = None
) -> dict:
    """
    Generate one complete file from a validated
    implementation plan.
    """

    spec = spec.strip()
    target_path = target_path.strip()

    if not spec:
        raise ValueError(
            "spec cannot be empty"
        )

    if not target_path:
        raise ValueError(
            "target_path cannot be empty"
        )

    validated_plan = (
        ImplementationPlan.model_validate(
            plan
        )
    )

    planned_paths = {
        file.path
        for file in validated_plan.files
    }

    if target_path not in planned_paths:
        raise ValueError(
            f"Target path is not planned: "
            f"{target_path}"
        )

    target_file = next(
        file
        for file in validated_plan.files
        if file.path == target_path
    )

    dependency_context = (
        dependency_context or {}
    )

    validate_dependency_context(
        target_file=target_file,
        dependency_context=dependency_context
    )

    user_payload = {
        "specification": spec,
        "implementation_plan":
            validated_plan.model_dump(),
        "target_path": target_path,
        "target_file_plan":
            target_file.model_dump(),
        "dependency_context":
            dependency_context
    }

    user_content = (
        "Generate exactly one file "
        "from this payload:\n\n"
        + json.dumps(
            user_payload,
            indent=2,
            ensure_ascii=False
        )
        + "\n\nCRITICAL: You must respond ONLY with a valid JSON object matching the GeneratedFile schema (keys: 'path', 'content', 'language', 'dependencies_used', 'assumptions'). Do not write markdown text or explanations outside the JSON object. Start your response directly with '{'."
    )
    if error_context:
        user_content += (
            f"\n\nCRITICAL: Your previous generation failed validation with error:\n"
            f"{error_context}\n"
            "You MUST ensure strict 4-space indentation for all code blocks (especially after 'with', 'for', 'if', 'def') "
            "and fix the syntax/format error shown above. Do not repeat the same broken formatting."
        )

    print(f"[ForgeCode]      -> Sending LLM request for {target_path}...", flush=True)
    content = await stream_chat_with_retries(
        client=client,
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "system",
                "content":
                    FILE_GENERATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.1 if not error_context else 0.25,
        max_retries=5,
        base_delay=2.0,
        log_prefix=f"[ForgeCode] [{target_path}]"
    )
    print(f"[ForgeCode]      -> LLM response received for {target_path}.", flush=True)


    if not content:
        raise ValueError(
            "Generator returned empty content"
        )

    raw_artifact = extract_json(
        content
    )

    raw_artifact = normalize_raw_artifact(
        raw_artifact=raw_artifact,
        target_path=target_path,
        target_file=target_file,
        dependency_context=dependency_context,
        validated_plan=validated_plan
    )

    try:
        artifact = GeneratedFile.model_validate(
            raw_artifact
        )

    except ValidationError as error:
        raise ValueError(
            f"Generated artifact failed validation: "
            f"{error}"
        ) from error

    validate_generated_artifact(
        artifact=artifact,
        target_path=target_path,
        planned_paths=planned_paths,
        plan=plan
    )

    return artifact.model_dump()


def validate_dependency_context(
    target_file,
    dependency_context: dict[str, str]
) -> None:
    """
    Ensure dependency context contains only
    dependencies declared by the target file.
    """

    allowed_dependencies = set(
        target_file.depends_on
    )

    provided_dependencies = set(
        dependency_context.keys()
    )

    unexpected = (
        provided_dependencies
        - allowed_dependencies
    )

    if unexpected:
        raise ValueError(
            "Dependency context contains "
            f"undeclared files: "
            f"{sorted(unexpected)}"
        )


def validate_generated_artifact(
    artifact: GeneratedFile,
    target_path: str,
    planned_paths: set[str],
    plan: dict | None = None
) -> None:
    """
    Enforce deterministic artifact consistency.
    """

    if artifact.path != target_path:
        raise ValueError(
            "Generated artifact path mismatch: "
            f"expected {target_path}, "
            f"got {artifact.path}"
        )

    allowed_external = {
        "torch", "torchvision", "torchaudio", "pytest", "numpy", "math",
        "os", "sys", "json", "typing", "pathlib", "re", "collections",
        "itertools", "functools", "time", "random", "unittest", "abc", "copy",
        "matplotlib", "pandas", "warning", "warnings", "traceback", "datetime", "uuid"
    }
    if plan and plan.get("dependencies"):
        for dep in plan["dependencies"]:
            allowed_external.add(str(dep).strip())

    # Common PyTorch and data science submodules/aliases
    common_submodules_aliases = {
        "nn", "F", "functional", "optim", "data", "dataloader", "utils",
        "transforms", "datasets", "models", "np", "pd", "plt"
    }

    # Normalize dependencies_used to match planned_paths where applicable
    normalized_deps = []
    for dep in artifact.dependencies_used:
        if dep in allowed_external or dep.split(".")[0] in allowed_external or dep in common_submodules_aliases:
            normalized_deps.append(dep)
            continue
        
        # Match against planned_paths
        normalized = dep
        if dep not in planned_paths:
            if f"{dep}.py" in planned_paths:
                normalized = f"{dep}.py"
            else:
                candidates = []
                dep_norm = dep.replace("\\", "/").lower()
                for p in planned_paths:
                    p_norm = p.replace("\\", "/").lower()
                    # Check if stem matches
                    stem = p_norm.rsplit(".", 1)[0]
                    if stem == dep_norm or stem.endswith("/" + dep_norm):
                        candidates.append(p)
                        continue
                    # Check if basename stem matches
                    basename = p_norm.split("/")[-1]
                    basename_stem = basename.rsplit(".", 1)[0]
                    if basename == dep_norm or basename_stem == dep_norm:
                        candidates.append(p)
                if len(candidates) == 1:
                    normalized = candidates[0]
        normalized_deps.append(normalized)
    
    artifact.dependencies_used = normalized_deps

    unknown_dependencies = set()
    for dep in artifact.dependencies_used:
        if dep in planned_paths or dep in allowed_external or dep in common_submodules_aliases:
            continue
        if dep.split(".")[0] in allowed_external:
            continue
        
        # Check if it matches any planned path stem
        dep_norm = dep.replace("\\", "/").lower()
        is_planned_stem = False
        for p in planned_paths:
            p_norm = p.replace("\\", "/").lower()
            stem = p_norm.rsplit(".", 1)[0]
            basename = p_norm.split("/")[-1]
            basename_stem = basename.rsplit(".", 1)[0]
            if stem == dep_norm or basename_stem == dep_norm:
                is_planned_stem = True
                break
        
        # If it doesn't match any planned path, doesn't end with .py, and has no slashes,
        # it is an external package/alias (like sklearn, tqdm, etc.) -> allow it.
        if not is_planned_stem and not dep.endswith(".py") and "/" not in dep and "\\" not in dep:
            continue
            
        unknown_dependencies.add(dep)

    if unknown_dependencies:
        raise ValueError(
            "Generated artifact references "
            f"unknown dependencies: "
            f"{sorted(unknown_dependencies)}"
        )

    validate_content_format(
        artifact
    )


def validate_content_format(
    artifact: GeneratedFile
) -> None:
    """
    Perform deterministic format validation.
    """

    suffix = (
        artifact.path
        .lower()
        .rsplit(".", 1)[-1]
        if "." in artifact.path
        else ""
    )

    if suffix == "json":
        try:
            json.loads(
                artifact.content
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Generated JSON content is invalid: "
                f"{error}"
            ) from error

    if suffix == "py":
        try:
            compile(
                artifact.content,
                artifact.path,
                "exec"
            )
        except SyntaxError as error:
            lines = artifact.content.splitlines()
            err_line = error.lineno or 1
            start_l = max(0, err_line - 3)
            end_l = min(len(lines), err_line + 2)
            snippet = "\n".join(f"{i+1}: {lines[i]}" for i in range(start_l, end_l))
            raise ValueError(
                f"Generated Python content has invalid syntax on line {err_line}: {error.msg}\n"
                f"Code snippet around line {err_line}:\n{snippet}"
            ) from error



def normalize_python_string_literals(
    content: str
) -> str:
    """
    Escape accidental raw newlines inside
    single- or double-quoted string literals.
    """

    result: list[str] = []
    i = 0
    n = len(content)
    in_string = False
    quote_char = ""
    is_triple = False
    escape = False

    while i < n:
        char = content[i]

        if not in_string:
            if char in ("'", '"'):
                triple = content[i:i + 3] == char * 3
                in_string = True
                quote_char = char
                is_triple = triple
                if triple:
                    result.append(char * 3)
                    i += 3
                    continue
            result.append(char)
            i += 1
            continue

        if escape:
            result.append(char)
            escape = False
            i += 1
            continue

        if char == "\\":
            result.append(char)
            escape = True
            i += 1
            continue

        if is_triple:
            if content[i:i + 3] == quote_char * 3:
                result.append(quote_char * 3)
                in_string = False
                quote_char = ""
                is_triple = False
                i += 3
                continue
            result.append(char)
            i += 1
            continue

        if char == "\n":
            result.append("\\n")
            i += 1
            continue

        result.append(char)
        if char == quote_char:
            in_string = False
            quote_char = ""
        i += 1

    return "".join(result)


def normalize_python_indentation(
    content: str,
    path: str = "<generated>"
) -> str:
    """
    Repair common LLM indentation collapse where block
    bodies use one or two spaces instead of 4-space levels.
    """

    try:
        compile(
            content,
            path,
            "exec"
        )
        return content
    except SyntaxError:
        pass

    lines = content.splitlines()
    repaired: list[str] = []
    level = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            repaired.append("")
            continue

        if stripped.startswith(("import ", "from ")):
            level = 0

        if stripped.startswith("class "):
            level = 0

        if stripped.startswith("def ") and level > 1:
            level = 1

        if stripped.startswith((
            "elif ",
            "else:",
            "except ",
            "except:",
            "finally:",
        )):
            level = max(0, level - 1)

        repaired.append(
            (" " * 4 * level) + stripped
        )

        if stripped.endswith(":"):
            level += 1

        if stripped.startswith(("return", "raise")) and level > 1:
            level -= 1

    repaired_content = "\n".join(repaired)
    if content.endswith("\n"):
        repaired_content += "\n"

    try:
        compile(
            repaired_content,
            path,
            "exec"
        )
        return repaired_content
    except SyntaxError:
        return content


def normalize_python_content(
    content: str,
    path: str = "<generated>"
) -> str:
    normalized = normalize_python_string_literals(
        content
    )
    return normalize_python_indentation(
        normalized,
        path=path
    )


def normalize_raw_artifact(
    raw_artifact,
    target_path: str,
    target_file,
    dependency_context: dict[str, str],
    validated_plan: ImplementationPlan,
) -> dict:

    if isinstance(raw_artifact, dict):
        content = (
            raw_artifact.get("content")
            or raw_artifact.get("code")
            or raw_artifact.get("source")
            or raw_artifact.get("source_code")
            or raw_artifact.get("file_content")
        )
        if content is not None:
            path = (
                raw_artifact.get("path")
                or raw_artifact.get("filename")
                or target_path
            )
            sfx = (
                target_path.lower().rsplit(".", 1)[-1]
                if "." in target_path
                else ""
            )
            language = raw_artifact.get("language") or (
                "python" if sfx == "py" else ("json" if sfx == "json" else "text")
            )
            normalized_content = str(content)
            if str(language).lower() == "python" or sfx == "py":
                normalized_content = normalize_python_content(
                    normalized_content,
                    path=str(path)
                )
            return {
                "path": str(path),
                "content": normalized_content,
                "language": str(language),
                "dependencies_used": list(
                    raw_artifact.get("dependencies_used")
                    or (target_file.depends_on if language != "json" else [])
                ),
                "assumptions": list(raw_artifact.get("assumptions") or [])
            }

    suffix = (
        target_path
        .lower()
        .rsplit(".", 1)[-1]
        if "." in target_path
        else ""
    )

    if suffix == "py":
        raise ValueError(
            f"Model returned unexpected output "
            f"for Python target {target_path}: "
            "refusing to fabricate source code"
        )

    if suffix == "json":
        content = json.dumps(
            raw_artifact,
            indent=2,
            ensure_ascii=False
        )
        language = "json"

    else:
        content = str(
            raw_artifact
        )
        language = "text"

    return {
        "path": target_path,
        "content": content,
        "language": language,
        "dependencies_used": (
            list(target_file.depends_on)
            if suffix != "json"
            else []
        ),
        "assumptions": []
    }

