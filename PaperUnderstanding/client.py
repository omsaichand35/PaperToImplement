import asyncio
import json
from collections import OrderedDict
from pathlib import Path

from openai import AsyncOpenAI

from paperforge_env import load_project_env

load_project_env()

from extractor import (
    build_numbered_text,
    extract_pdf_pages
)
from models import (
    Evidence,
    ImplementationFact,
    ImplementationSpec,
    ModelComponent
)


MODEL_NAME = "meta/llama-3.1-70b-instruct"

PAGES_PER_CHUNK = 4

client = AsyncOpenAI(
    base_url=(
        "https://integrate.api.nvidia.com/v1"
    ),
    max_retries=3,
    timeout=300.0
)


SYSTEM_PROMPT = """
You are a research-paper implementation analyst.

Your task is to convert a deep-learning research
paper into a structured implementation specification.

You are NOT writing code.

Extract only implementation-relevant information.

Important categories:

1. model architecture
2. model components
3. layer counts
4. channel dimensions
5. kernel sizes
6. stride
7. padding
8. dilation
9. activation functions
10. normalization
11. preprocessing
12. input shapes
13. output shapes
14. loss functions
15. optimizer
16. learning rate
17. batch size
18. epochs
19. schedulers
20. referenced external methods

Evidence rules:

- PAPER_REPORTED:
  explicitly stated by the paper

- INFERRED:
  logically inferred but not explicitly stated

- ASSUMED:
  implementation choice not established by paper

- UNKNOWN:
  required detail not found

- AMBIGUOUS:
  conflicting or unclear information

- REFERENCED_ELSEWHERE:
  paper delegates the method to another cited work

Never convert missing information into certainty.

Every PAPER_REPORTED fact should include page-level
evidence whenever possible.

Return JSON only.

CRITICAL: You must return a JSON object populated with ACTUAL EXTRACTED VALUES from the paper chunk.
Do NOT copy the schema definition objects (such as {"anyOf": ..., "type": ...} or {"$ref": ...}) into the values!
For example, paper_title must be a plain string (or null), and model_components must be a list of actual component objects.
The JSON must match the requested schema instance exactly.
"""


def build_schema_prompt() -> str:

    schema = (
        ImplementationSpec
        .model_json_schema()
    )

    return json.dumps(
        schema,
        indent=2
    )


def split_pages_into_chunks(
    pages: list[dict],
    pages_per_chunk: int = PAGES_PER_CHUNK
) -> list[list[dict]]:

    return [
        pages[index:index + pages_per_chunk]
        for index in range(
            0,
            len(pages),
            pages_per_chunk
        )
    ]


def extract_json_payload(
    content: str
) -> dict:
    def is_valid_payload(obj) -> bool:
        if isinstance(obj, dict):
            return True
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return True
        return False

    text = content.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            lines = part.strip().splitlines()
            if lines and lines[0].strip().lower() in {"json", "python", "text", "sh", "bash", "yaml", "markdown"}:
                lines = lines[1:]
            candidate = "\n".join(lines).strip()
            try:
                obj = json.loads(candidate)
                if is_valid_payload(obj):
                    return obj
            except Exception:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(candidate)
                    if is_valid_payload(obj):
                        return obj
                except Exception:
                    continue

    try:
        obj = json.loads(text)
        if is_valid_payload(obj):
            return obj
    except Exception:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text)
            if is_valid_payload(obj):
                return obj
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for idx in range(len(text)):
        if text[idx] in "{[":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if is_valid_payload(obj):
                    return obj
            except Exception:
                continue

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "LLM response did not contain a JSON object"
        )

    return json.loads(
        text[start:end + 1]
    )


def fact_key(
    fact: ImplementationFact
) -> tuple:

    evidence_key = tuple(
        (
            item.page,
            item.section,
            item.quote
        )
        for item in fact.evidence
    )

    value_key = json.dumps(
        fact.value,
        sort_keys=True,
        ensure_ascii=False,
        default=str
    )

    return (
        fact.name,
        value_key,
        fact.status,
        evidence_key,
        fact.notes
    )


def merge_fact_lists(
    existing: list[ImplementationFact],
    incoming: list[ImplementationFact]
) -> list[ImplementationFact]:

    merged = OrderedDict()

    for fact in existing + incoming:
        merged[fact_key(fact)] = fact

    return list(merged.values())


def merge_components(
    existing: list[ModelComponent],
    incoming: list[ModelComponent]
) -> list[ModelComponent]:

    merged: OrderedDict[
        tuple[str, str | None],
        ModelComponent
    ] = OrderedDict()

    for component in existing + incoming:
        key = (
            component.name,
            component.component_type
        )

        if key not in merged:
            merged[key] = ModelComponent(
                name=component.name,
                component_type=component.component_type,
                facts=list(component.facts)
            )
            continue

        merged[key].facts = merge_fact_lists(
            merged[key].facts,
            component.facts
        )

    return list(merged.values())


def merge_specs(
    specs: list[ImplementationSpec]
) -> ImplementationSpec:

    merged = ImplementationSpec()

    for spec in specs:

        if not merged.paper_title and spec.paper_title:
            merged.paper_title = spec.paper_title

        if not merged.task and spec.task:
            merged.task = spec.task

        merged.model_components = merge_components(
            merged.model_components,
            spec.model_components
        )

        merged.preprocessing = merge_fact_lists(
            merged.preprocessing,
            spec.preprocessing
        )

        merged.training = merge_fact_lists(
            merged.training,
            spec.training
        )

        merged.unknowns = merge_fact_lists(
            merged.unknowns,
            spec.unknowns
        )

    # --- Deduplicate unknowns ---
    # Collect all fact names that were resolved as PAPER_REPORTED
    # across model_components, preprocessing, and training.
    resolved_names: set[str] = set()
    for comp in merged.model_components:
        for fact in comp.facts:
            if fact.status == "PAPER_REPORTED":
                resolved_names.add(fact.name.lower().strip())
    for fact in merged.preprocessing:
        if fact.status == "PAPER_REPORTED":
            resolved_names.add(fact.name.lower().strip())
    for fact in merged.training:
        if fact.status == "PAPER_REPORTED":
            resolved_names.add(fact.name.lower().strip())

    # Filter unknowns: drop any whose name (case-insensitive) or whose
    # subject is a substring of a resolved fact name.
    filtered_unknowns = []
    for unknown in merged.unknowns:
        u_name = unknown.name.lower().strip()
        # Check exact match
        if u_name in resolved_names:
            continue
        # Check substring overlap (e.g. "dropout_rate" vs "dropout probability")
        u_words = set(u_name.replace("_", " ").split())
        skip = False
        for resolved in resolved_names:
            r_words = set(resolved.replace("_", " ").split())
            # If >50% of the unknown's key words appear in a resolved name
            if len(u_words & r_words) >= max(1, len(u_words) // 2):
                skip = True
                break
        if not skip:
            filtered_unknowns.append(unknown)

    merged.unknowns = filtered_unknowns

    return merged


async def stream_chat_with_retries(
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_retries: int = 5,
    base_delay: float = 2.0,
    log_prefix: str = "[PaperUnderstanding]"
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


async def analyze_chunk(
    pages: list[dict],
    schema: str,
    chunk_index: int,
    total_chunks: int
) -> ImplementationSpec:

    paper_text = build_numbered_text(
        pages
    )

    user_prompt = f"""
Analyze the following research paper chunk.

This is chunk {chunk_index} of {total_chunks}.

Build an implementation specification.

Important chunking rule:
If this chunk does not mention a detail, omit it
instead of marking it UNKNOWN. Only emit UNKNOWN
or REFERENCED_ELSEWHERE when this chunk provides
evidence for that judgment.

CRITICAL: Return a JSON INSTANCE with actual strings, numbers, and lists extracted from the text below. Do NOT copy the schema definitions!
SCHEMA:

{schema}

PAPER CHUNK:

{paper_text}
"""

    print(f"[PaperUnderstanding]   -> Sending LLM request for chunk {chunk_index}/{total_chunks}...", flush=True)
    content = await stream_chat_with_retries(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_retries=5,
        base_delay=2.0,
        log_prefix=f"[PaperUnderstanding] [Chunk {chunk_index}/{total_chunks}]"
    )
    print(f"[PaperUnderstanding]   -> Finished reading chunk {chunk_index}/{total_chunks} ({len(content)} chars).", flush=True)

    if not content:
        raise ValueError(
            "LLM returned empty content"
        )

    data = extract_json_payload(
        content
    )
    if isinstance(data, dict) and "model_components" not in data and "paper_title" not in data:
        for val in data.values():
            if isinstance(val, dict) and ("model_components" in val or "paper_title" in val):
                data = val
                break
    print(f"[PaperUnderstanding]   -> Extracted keys for chunk {chunk_index}: {list(data.keys()) if isinstance(data, dict) else type(data)}, components count: {len(data.get('model_components', [])) if isinstance(data, dict) else 0}", flush=True)

    return (
        ImplementationSpec
        .model_validate(data)
    )


async def analyze_paper(
    pdf_path: str
) -> ImplementationSpec:
    print(f"[PaperUnderstanding] Analyzing PDF: {pdf_path}", flush=True)
    pages = extract_pdf_pages(
        pdf_path
    )

    schema = build_schema_prompt()
    page_chunks = split_pages_into_chunks(
        pages
    )

    print(f"[PaperUnderstanding] Split into {len(page_chunks)} chunks. Analyzing sequentially to ensure API stability...", flush=True)
    chunk_specs = []
    for index, chunk in enumerate(page_chunks, start=1):
        spec = await analyze_chunk(
            pages=chunk,
            schema=schema,
            chunk_index=index,
            total_chunks=len(page_chunks)
        )
        chunk_specs.append(spec)
    print(f"[PaperUnderstanding] All {len(page_chunks)} chunks analyzed successfully. Merging specs...", flush=True)

    return merge_specs(
        chunk_specs
    )


async def main():

    pdf_path = (
        Path(__file__).parent
        / "papers"
        / "NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf"
    )

    spec = await analyze_paper(
        str(pdf_path)
    )

    output_path = (
        Path(__file__).parent
        / "outputs"
        / "implementation_spec.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        spec.model_dump_json(
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nImplementation specification created"
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
