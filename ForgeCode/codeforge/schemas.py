import ast
import asyncio
import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator
)


class SafeBaseModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def coerce_none_collections(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in list(data.items()):
                if k in cls.model_fields:
                    f_info = cls.model_fields[k]
                    annot = str(f_info.annotation) if f_info.annotation else ""
                    if "list" in annot or "List" in annot:
                        if v is None:
                            data[k] = []
                        elif not isinstance(v, (list, dict)):
                            data[k] = [v]
                    elif "dict" in annot or "Dict" in annot:
                        if v is None:
                            data[k] = {}
        return data


class PlannedFile(SafeBaseModel):
    path: str = Field(
        description=(
            "Project-relative file path"
        )
    )

    purpose: str = Field(
        description=(
            "Why this file is required"
        )
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete responsibilities "
            "of this file"
        )
    )

    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Other planned project files "
            "this file depends on"
        )
    )


class ImplementationPlan(SafeBaseModel):

    project_name: str

    framework: str

    task_type: str

    summary: str

    dependencies: list[str] = Field(
        default_factory=list
    )

    files: list[PlannedFile]

    implementation_order: list[str]

    assumptions: list[str] = Field(
        default_factory=list
    )

    unresolved_questions: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_plan_consistency(self):

        # Reject duplicate file paths
        file_paths = [
            file.path
            for file in self.files
        ]

        if len(file_paths) != len(
            set(file_paths)
        ):
            raise ValueError(
                "files contains duplicate paths"
            )

        planned_paths = set(file_paths)

        # Reject duplicate order entries
        if len(self.implementation_order) != len(
            set(self.implementation_order)
        ):
            raise ValueError(
                "implementation_order contains "
                "duplicates"
            )

        # implementation_order must not reference
        # unknown paths
        unknown_order_paths = (
            set(self.implementation_order)
            - planned_paths
        )

        if unknown_order_paths:
            raise ValueError(
                "implementation_order contains "
                "paths not in files: "
                f"{sorted(unknown_order_paths)}"
            )

        # implementation_order must cover every
        # planned file — no silent omissions
        missing_order_paths = (
            planned_paths
            - set(self.implementation_order)
        )

        if missing_order_paths:
            raise ValueError(
                "implementation_order is missing "
                "planned paths: "
                f"{sorted(missing_order_paths)}"
            )

        # Every depends_on must reference a known planned file
        for file in self.files:
            file.depends_on = [
                dep for dep in file.depends_on if dep in planned_paths and dep != file.path
            ]

        return self


class GeneratedFile(SafeBaseModel):
    path: str = Field(
        description="Project-relative target file path"
    )

    content: str = Field(
        description="Complete generated file content"
    )

    language: str = Field(
        description=(
            "Language or format such as "
            "python, json, yaml, markdown"
        )
    )

    dependencies_used: list[str] = Field(
        default_factory=list,
        description=(
            "Planned project files or external "
            "dependencies actually used"
        )
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Assumptions introduced while "
            "generating this file"
        )
    )


class ReviewIssue(SafeBaseModel):
    severity: Literal[
        "critical",
        "error",
        "warning",
        "info"
    ]

    category: Literal[
        "missing_requirement",
        "invented_detail",
        "cross_file_inconsistency",
        "dependency_mismatch",
        "interface_mismatch",
        "plan_deviation",
        "other"
    ]

    message: str

    affected_files: list[str] = Field(
        default_factory=list
    )

    evidence: list[str] = Field(
        default_factory=list
    )

    recommendation: str | None = None


class ImplementationReview(SafeBaseModel):
    passed: bool

    summary: str

    issues: list[ReviewIssue] = Field(
        default_factory=list
    )

    checked_files: list[str] = Field(
        default_factory=list
    )

    missing_requirements: list[str] = Field(
        default_factory=list
    )

    invented_details: list[str] = Field(
        default_factory=list
    )

    cross_file_inconsistencies: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_review_consistency(self):

        blocking_issues = {
            "critical",
            "error"
        }

        has_blocking_issue = any(
            issue.severity in blocking_issues
            for issue in self.issues
        )

        if self.passed and has_blocking_issue:
            raise ValueError(
                "Review cannot pass while critical "
                "or error issues exist"
            )

        return self


def extract_json(content: str) -> Any:
    """
    Extract JSON from plain or fenced output, rejecting primitive lists
    like [2, 4, 6, 8] that appear in LLM reasoning text.
    Handles trailing commas, Python booleans/None, single quotes, and line comments.
    """
    def is_valid_payload(obj: Any) -> bool:
        if isinstance(obj, dict):
            return True
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return True
        return False

    def clean_json_str(text: str) -> str:
        lines = []
        for line in text.splitlines():
            l = line.strip()
            if l.startswith("//") or (l.startswith("#") and not l.startswith("# [")):
                continue
            if "//" in line and not "://" in line:
                line = line.split("//")[0]
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
        cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
        cleaned = re.sub(r'\bNone\b', 'null', cleaned)
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
        return cleaned.strip()

    def find_balanced_blocks(text: str) -> list[str]:
        blocks = []
        stack = []
        start_idx = -1
        in_string = False
        escape = False
        quote_char = None

        for idx, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None
                continue
            if in_string:
                continue

            if char in ('{', '['):
                if not stack:
                    start_idx = idx
                stack.append(char)
            elif char in ('}', ']'):
                if stack:
                    top = stack[-1]
                    if (char == '}' and top == '{') or (char == ']' and top == '['):
                        stack.pop()
                        if not stack and start_idx != -1:
                            blocks.append(text[start_idx:idx+1])
                            start_idx = -1
                    else:
                        stack.clear()
                        start_idx = -1
        return blocks

    content_clean = content.strip()

    if "```" in content_clean:
        parts = content_clean.split("```")
        for part in parts[1:]:
            lines = part.strip().splitlines()
            if lines and lines[0].strip().lower() in {
                "json", "python", "text", "sh", "bash", "yaml", "markdown"
            }:
                lines = lines[1:]
            candidate = "\n".join(lines).strip()
            for cand in (candidate, clean_json_str(candidate)):
                try:
                    obj = json.loads(cand)
                    if is_valid_payload(obj):
                        return obj
                except Exception:
                    try:
                        obj, _ = json.JSONDecoder().raw_decode(cand)
                        if is_valid_payload(obj):
                            return obj
                    except Exception:
                        try:
                            obj = ast.literal_eval(cand)
                            if is_valid_payload(obj):
                                return obj
                        except Exception:
                            continue

    for cand in (content_clean, clean_json_str(content_clean)):
        try:
            obj = json.loads(cand)
            if is_valid_payload(obj):
                return obj
        except Exception:
            try:
                obj, _ = json.JSONDecoder().raw_decode(cand)
                if is_valid_payload(obj):
                    return obj
            except Exception:
                try:
                    obj = ast.literal_eval(cand)
                    if is_valid_payload(obj):
                        return obj
                except Exception:
                    pass

    for block in find_balanced_blocks(content_clean):
        for cand in (block, clean_json_str(block)):
            try:
                obj = json.loads(cand)
                if is_valid_payload(obj):
                    return obj
            except Exception:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(cand)
                    if is_valid_payload(obj):
                        return obj
                except Exception:
                    try:
                        obj = ast.literal_eval(cand)
                        if is_valid_payload(obj):
                            return obj
                    except Exception:
                        continue

    decoder = json.JSONDecoder()
    for cand in (content_clean, clean_json_str(content_clean)):
        for idx in range(len(cand)):
            if cand[idx] in "{[":
                try:
                    obj, _ = decoder.raw_decode(cand[idx:])
                    if is_valid_payload(obj):
                        return obj
                except Exception:
                    try:
                        obj = ast.literal_eval(cand[idx:])
                        if is_valid_payload(obj):
                            return obj
                    except Exception:
                        continue

    print(f"\n[ForgeCode] [extract_json] ERROR: Could not extract valid JSON dict/list from content!\n--- RAW CONTENT (first 1500 chars) ---\n{content[:1500]}\n--------------------------------------\n", flush=True)
    return json.loads(content)


async def stream_chat_with_retries(
    client: Any,
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_retries: int = 5,
    base_delay: float = 2.0,
    log_prefix: str = "[ForgeCode]"
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await (
                client
                .chat
                .completions
                .create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )
            )
            chunks = []
            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                chunks.append(delta)
            return "".join(chunks)
        except Exception as exc:
            if attempt == max_retries:
                print(f"{log_prefix}   -> All {max_retries} attempts failed: {exc}. Raising error.", flush=True)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"{log_prefix}   -> Error during LLM streaming ({exc}). Retrying in {delay}s (attempt {attempt}/{max_retries})...", flush=True)
            await asyncio.sleep(delay)