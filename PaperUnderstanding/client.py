from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so paperforge_env can be imported.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

# pyrefly: ignore [missing-import]
from openai import AsyncOpenAI
from pydantic import ValidationError

from paperforge_env import load_project_env

from extractor import (
    build_numbered_text,
    extract_pdf_pages,
    split_pages_into_chunks,
)
from models import (
    Evidence,
    ImplementationFact,
    ImplementationSpec,
    ModelComponent,
    ArchitectureGraph,
)


load_project_env()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "meta/llama-3.1-70b-instruct"
VISION_MODEL_NAME = "meta/llama-3.2-90b-vision-instruct"
ENABLE_VISION_EXTRACTION = True

PAGES_PER_CHUNK = 4
OVERLAP_PAGES = 1

EXTRACTION_TEMPERATURE = 0.1
SYNTHESIS_TEMPERATURE = 0.0
VERIFICATION_TEMPERATURE = 0.0

MAX_RETRIES = 5
BASE_RETRY_DELAY = 2.0

MAX_CONCURRENT_REQUESTS = 1

OUTPUT_FILENAME = "implementation_spec.json"

VALID_STATUSES = {
    "PAPER_REPORTED",
    "INFERRED",
    "ASSUMED",
    "UNKNOWN",
    "AMBIGUOUS",
    "REFERENCED_ELSEWHERE",
    "REGISTRY_CANONICAL",
    "LITERATURE_GROUNDED",
    "DOMAIN_HEURISTIC",
}


client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    max_retries=3,
    timeout=300.0,
)


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are a research-paper implementation evidence extractor.

Your task is NOT to write code.
Your task is NOT to guess a complete architecture from incomplete text.
Your task is to extract implementation-relevant claims from the supplied
paper chunk with precise evidence.

Extract implementation-relevant information including:

1. model architecture
2. model components
3. forward-pass operations
4. connections between components
5. layer counts
6. hidden/channel/embedding dimensions
7. tensor shapes
8. patch sizes
9. kernel sizes
10. stride
11. padding
12. dilation
13. activation functions
14. normalization
15. conditioning mechanisms
16. fusion mechanisms
17. preprocessing
18. input shapes
19. output shapes
20. loss functions
21. optimizer
22. learning rates
23. batch size
24. epochs
25. schedulers
26. inference procedure
27. model variants
28. referenced external methods

Evidence statuses:

- PAPER_REPORTED:
  The claim is explicitly stated by the supplied paper text.

- INFERRED:
  The claim follows logically from explicit paper statements but is not
  itself directly stated.

- ASSUMED:
  The claim is an implementation choice not established by the paper.

- UNKNOWN:
  The chunk explicitly establishes that a required detail is unavailable
  or unspecified. Do not emit UNKNOWN merely because this chunk omits it.

- AMBIGUOUS:
  The supplied text contains conflicting or genuinely unclear information.

- REFERENCED_ELSEWHERE:
  The paper explicitly delegates the detail to another cited method or work.

Critical evidence rules:

1. Never convert missing information into certainty.
2. Never infer a numeric value merely because that number appears near a
   related term.
3. A denominator, normalization constant, reference batch size, equation
   constant, ablation value, or metric value is NOT automatically a model
   hyperparameter.
4. Example:
      lr = blr * global_batch / 256
   does NOT imply:
      batch_size = 256
5. Example:
      image height is 512
   does NOT automatically imply:
      universal input_shape = 512 x 512
6. Distinguish:
      base learning rate
      effective learning rate
      learning-rate scaling rule
7. Distinguish:
      prediction parameterization
      optimization target
      loss function
8. Distinguish:
      training resolution
      evaluation resolution
      universal model input shape
9. Distinguish:
      spatial output resolution
      full output tensor shape
10. Every PAPER_REPORTED fact must contain evidence whenever possible.
11. Evidence must directly support the claim.
12. Keep exact tensor relationships when explicitly reported.
13. Preserve architectural operations such as:
      concatenation
      projection
      patchification
      unpatchification
      addition
      cross-attention
      self-attention
      conditioning
      reshaping
      permutation
14. If the text says A and B are concatenated into C, do not extract only
    A, B, and C independently. Preserve the relationship as an
    implementation-relevant fact.
15. Do not use knowledge from outside the supplied chunk.

Return JSON only.

CRITICAL:
Return a JSON INSTANCE containing actual extracted values.
Do NOT copy JSON Schema definition objects into values.
"""


VISION_EXTRACTION_SYSTEM_PROMPT = """
You are a research-paper visual architecture and implementation evidence extractor.

Your task is to examine research-paper architecture diagrams, block diagrams, flow figures, and tables depicted in visual figure images, alongside accompanying captions/text, to extract precise implementation facts.

Extract visual architecture evidence including:
1. Model architecture overview and overall pipeline flow
2. Components and submodules depicted in the figure (encoders, decoders, backbones, heads, attention blocks, MLPs, projection layers)
3. Layer order and sequential/parallel forward-pass flow shown by arrows and blocks
4. Skip connections, residual connections, and branching paths
5. Tensor shapes, channel dimensions, spatial resolutions, and embedding sizes annotated on block diagrams
6. Patchification, tokenization, pooling, upsampling/downsampling operations
7. Normalization layers (LayerNorm, BatchNorm, RMSNorm) and Activation functions shown in diagrams
8. Conditioning pathways (e.g. timestep embedding, class label conditioning, cross-attention inputs)

Evidence rules for visual extraction:
1. Treat visual figures as the PRIMARY TOPOLOGY SOURCE. Figures often contain arrows, branches, skip connections, and residual connections that text never mentions.
2. Every PAPER_REPORTED fact must include evidence referencing what is shown in the figure (e.g. quote="Figure 2 architecture diagram shows input tensor [B, C, H, W] passed to PatchEmbed block").
3. Preserve connected operations, skip connections, residuals, branches, and block sequence depicted by arrows or block layout.
4. Do not invent dimensions or hyperparameters if they are not legible or annotated in the figure.

Return JSON only matching the schema.

CRITICAL:
Return a JSON INSTANCE containing actual extracted values.
Do NOT copy JSON Schema definition objects into values.
"""


SYNTHESIS_SYSTEM_PROMPT = """
You are a senior research-paper architecture reconstruction analyst.

You receive:

1. evidence extracted from overlapping chunks of one paper
2. the paper's implementation-specification schema

Your task is to reconstruct one globally coherent implementation
specification.

This is a synthesis task, not a mechanical merge.

Primary objective:
Reconstruct the end-to-end implementation story while preserving only
claims supported by the extracted evidence.

You must:

1. Merge semantically identical components even when chunk extractors used
   different names or component_type labels.

2. Merge semantically identical facts such as:
      optimizer / Optimizer
      epochs / Number of epochs
   while preserving distinct concepts such as:
      base learning rate / effective learning rate
      input shape / training resolution
      x-prediction / velocity-space loss

3. Reconstruct cross-chunk architecture relationships.

4. Preserve forward-pass order whenever evidence supports it.

5. Preserve tensor transformations whenever evidence supports them.

6. Preserve model variants separately when their configurations differ.

7. Detect conflicts instead of silently choosing one value.

8. Never promote weak inference to PAPER_REPORTED.

9. Never infer:
      batch_size = 256
   from:
      lr = blr * global_batch / 256

10. Never infer a universal input shape from a stage-specific resolution.

11. Distinguish:
      model architecture
      preprocessing
      training
      inference

12. Remove duplicates.

13. If two facts conflict:
      - use AMBIGUOUS when unresolved
      - preserve evidence for both sides in notes/evidence where possible

14. Do not invent missing timestep-conditioning mechanisms, layer counts,
    hidden dimensions, or other implementation details.

15. If a detail is required for implementation but the evidence does not
    establish it, preserve it as UNKNOWN only when the combined evidence
    justifies identifying it as unresolved.

Architecture reconstruction rule:

For each major path, reason in terms of:

    input
      -> transformation
      -> output
      -> next transformation

When evidence supports relationships, preserve them in facts rather than
returning only disconnected labels.

Return JSON only.

CRITICAL:
Return a JSON INSTANCE containing actual values.
Do NOT return schema definitions.
"""


VERIFICATION_SYSTEM_PROMPT = """
You are a strict claim-evidence verifier for research-paper implementation
specifications.

You receive:

1. a synthesized implementation specification
2. numbered paper text
3. the required JSON schema

Your job is to correct unsupported certainty.

For every fact, verify:

A. Does the evidence directly support the claim?
B. Does the numeric value mean what the fact name claims?
C. Is the claim explicit, inferred, ambiguous, or unsupported?
D. Is the evidence from the correct page and context?
E. Has a stage-specific value been incorrectly generalized?
F. Has an equation constant been mistaken for a hyperparameter?
G. Has a reference scaling denominator been mistaken for batch size?
H. Has an ablation-table value been mistaken for the final configuration?
I. Has prediction parameterization been confused with optimization loss?
J. Has spatial resolution been confused with full tensor shape?

Required behavior:

- Keep supported PAPER_REPORTED facts.
- Downgrade indirect facts to INFERRED.
- Mark conflicting facts AMBIGUOUS.
- Remove claims that are unsupported and not useful.
- Use UNKNOWN only for genuinely unresolved implementation details.
- Correct duplicate facts.
- Correct duplicate components.
- Preserve direct evidence.
- Do not invent new facts.
- Do not use external knowledge.

Specific anti-hallucination rule:

The statement:
    lr = blr * global_batch / 256
does NOT support:
    batch_size = 256

The statement:
    Stage 2 uses image height 512
does NOT by itself support:
    universal input_shape = 512 x 512

Return the corrected full JSON specification only.
"""


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

def build_schema_prompt() -> str:
    schema = ImplementationSpec.model_json_schema()

    return json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------

def _is_valid_json_payload(
    obj: Any
) -> bool:
    if isinstance(obj, dict):
        return True

    if (
        isinstance(obj, list)
        and all(isinstance(item, dict) for item in obj)
    ):
        return True

    return False


def extract_json_payload(
    content: str
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Extract the first plausible JSON payload from an LLM response.

    Supports:
    - plain JSON
    - fenced JSON
    - leading/trailing prose
    - raw-decoded JSON prefixes
    """
    if not content:
        raise ValueError(
            "LLM response was empty"
        )

    text = content.strip()

    # -------------------------------------------------------------
    # 1. Try fenced code blocks first.
    # -------------------------------------------------------------
    if "```" in text:
        parts = text.split("```")

        for part in parts[1:]:
            lines = part.strip().splitlines()

            if (
                lines
                and lines[0].strip().lower()
                in {
                    "json",
                    "python",
                    "text",
                    "sh",
                    "bash",
                    "yaml",
                    "markdown",
                }
            ):
                lines = lines[1:]

            candidate = "\n".join(lines).strip()

            if not candidate:
                continue

            try:
                obj = json.loads(candidate)

                if _is_valid_json_payload(obj):
                    return obj

            except Exception:
                try:
                    obj, _ = (
                        json.JSONDecoder()
                        .raw_decode(candidate)
                    )

                    if _is_valid_json_payload(obj):
                        return obj

                except Exception:
                    continue

    # -------------------------------------------------------------
    # 2. Try entire response.
    # -------------------------------------------------------------
    try:
        obj = json.loads(text)

        if _is_valid_json_payload(obj):
            return obj

    except Exception:
        pass

    # -------------------------------------------------------------
    # 3. Try raw decode from start.
    # -------------------------------------------------------------
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)

        if _is_valid_json_payload(obj):
            return obj

    except Exception:
        pass

    # -------------------------------------------------------------
    # 4. Scan for a JSON object/array start.
    # -------------------------------------------------------------
    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char not in "{[":
            continue

        try:
            obj, _ = decoder.raw_decode(
                text[index:]
            )

            if _is_valid_json_payload(obj):
                return obj

        except Exception:
            continue

    raise ValueError(
        "LLM response did not contain a valid JSON payload"
    )


def unwrap_spec_payload(
    data: Any
) -> Any:
    """
    Handle occasional wrapper objects such as:

    {
        "implementation_spec": {
            ...
        }
    }
    """
    if not isinstance(data, dict):
        return data

    if (
        "model_components" in data
        or "paper_title" in data
        or "task" in data
    ):
        return data

    for value in data.values():
        if not isinstance(value, dict):
            continue

        if (
            "model_components" in value
            or "paper_title" in value
            or "task" in value
        ):
            return value

    return data


# ---------------------------------------------------------------------
# LLM communication
# ---------------------------------------------------------------------

async def stream_chat_with_retries(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.1,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_RETRY_DELAY,
    log_prefix: str = "[PaperUnderstanding]",
) -> str:
    """
    Stream an LLM response with exponential-backoff retries.
    """
    for attempt in range(
        1,
        max_retries + 1
    ):
        try:
            response = await (
                client
                .chat
                .completions
                .create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
            )

            chunks: list[str] = []

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = (
                    chunk.choices[0]
                    .delta
                    .content
                    or ""
                )

                chunks.append(delta)

            content = "".join(chunks).strip()

            if not content:
                raise ValueError(
                    "LLM returned empty content"
                )

            return content

        except Exception as exc:
            if attempt == max_retries:
                print(
                    f"{log_prefix} "
                    f"All {max_retries} attempts failed: "
                    f"{exc}",
                    flush=True,
                )
                raise

            delay = (
                base_delay
                * (2 ** (attempt - 1))
            )

            print(
                f"{log_prefix} "
                f"Error during LLM streaming: {exc}. "
                f"Retrying in {delay}s "
                f"(attempt {attempt}/{max_retries})...",
                flush=True,
            )

            await asyncio.sleep(delay)

    raise RuntimeError(
        "Unreachable retry state"
    )


# ---------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------

# Fields where every list element must be a plain str
_STR_LIST_FIELDS = frozenset({
    "inputs", "outputs", "consumers", "initialization", "component_ids",
})

# Fields where every list element must be an ArchitectureOperation-compatible dict
_OPERATION_LIST_FIELDS = frozenset({
    "operations",
})

# Fields that must be a single dict/object — if LLM emits a list, fold into .facts
_SINGLETON_OBJECT_FIELDS = frozenset({
    "inference", "architecture", "architecture_graph", "losses", "hardware", "reproducibility",
})

# Fields that expect a list of ImplementationFacts
_FACT_LIST_FIELDS = frozenset({
    "training", "preprocessing", "unknowns", "facts"
})

# Fields that expect a list of ModelComponents
_COMPONENT_LIST_FIELDS = frozenset({
    "model_components"
})


def _dict_to_fact_list(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce a dictionary to a list of ImplementationFact dicts."""
    facts = []
    for k, v in d.items():
        if isinstance(v, dict) and ("value" in v or "status" in v):
            fact = {"name": k}
            fact.update(v)
            facts.append(fact)
        else:
            facts.append({
                "name": k,
                "value": v,
                "status": "PAPER_REPORTED",
            })
    return facts


def _dict_to_component_list(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce a dictionary to a list of ModelComponent dicts."""
    components = []
    for k, v in d.items():
        if isinstance(v, dict):
            comp = {"name": k}
            comp.update(v)
            components.append(comp)
        else:
            components.append({
                "name": k,
                "component_type": "OTHER",
                "facts": [],
            })
    return components


def _item_to_str(item: Any) -> str:
    """Convert one list element to a plain string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        name = item.get("name") or item.get("id")
        dims = item.get("dimensions") or item.get("shape")
        if name:
            return f"{name} {dims}" if dims else str(name)
        return str(item)
    return str(item)


def _str_to_op_dict(name: str) -> dict[str, Any]:
    """Upgrade a bare string into a minimal ArchitectureOperation dict."""
    op_id = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "op"
    return {
        "id": op_id,
        "name": name,
        "operation_type": "UNKNOWN",
    }


def _str_to_node_dict(name: str) -> dict[str, Any]:
    """Upgrade a bare string into a minimal ArchitectureNode dict."""
    node_id = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "node"
    return {
        "id": node_id,
        "name": name,
        "type": "Unknown",
    }


def _str_to_edge_dict(spec_str: str) -> dict[str, Any]:
    """Upgrade a bare string like 'A -> B' into a minimal ArchitectureEdge dict."""
    parts = re.split(r"\s+(?:->|to)\s+", spec_str, maxsplit=1)
    if len(parts) == 2:
        return {"from": parts[0].strip(), "to": parts[1].strip()}
    return {"from": spec_str.strip(), "to": spec_str.strip()}


def _str_to_forward_pass_step_dict(spec_str: str, step_idx: int = 1) -> dict[str, Any]:
    """Upgrade a bare string like 'memory = self.encoder(embedded_tokens)' into a ForwardPassStep dict."""
    match = re.match(
        r"^(?P<output>[A-Za-z0-9_, ]+)\s*=\s*(?:self\.)?(?P<operation>[A-Za-z0-9_]+)\s*\((?P<input>.*)\)\s*$",
        spec_str.strip(),
    )
    if match:
        return {
            "step": step_idx,
            "operation": match.group("operation"),
            "input": match.group("input").strip(),
            "output": match.group("output").strip(),
        }
    return {
        "step": step_idx,
        "operation": spec_str.strip(),
        "input": "x",
        "output": "out",
    }


def _list_to_singleton_object(items: list[Any], key: str) -> dict[str, Any]:
    """
    Fold a list emitted by the LLM for a singleton-object field into a minimal
    compatible dict by placing ImplementationFact-like items into .facts.
    """
    facts: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            # Looks like an ImplementationFact — keep it
            if "name" in item and ("value" in item or "status" in item):
                facts.append(item)
            else:
                # Unknown dict structure — wrap as a fact
                facts.append({
                    "name": item.get("name", str(item)),
                    "value": str(item),
                    "status": "UNKNOWN",
                })
        elif isinstance(item, str):
            facts.append({
                "name": item,
                "value": None,
                "status": "UNKNOWN",
            })
    return {"facts": facts}


def _str_to_executable_op_dict(s: str) -> dict[str, Any]:
    text = s.strip()
    if ":" in text:
        parts = text.split(":", 1)
        return {"operation": parts[0].strip(), "formula": parts[1].strip()}
    if "=" in text:
        parts = text.split("=", 1)
        return {"operation": parts[0].strip(), "formula": parts[1].strip()}
    return {"operation": "operation", "formula": text}


def _sanitize_spec_payload_dict(data: Any, _parent_key: str = "") -> Any:
    """
    Bidirectional coercion before Pydantic validation:

    * str-list fields (inputs, outputs, …): dict items → str
    * operation-list fields (operations):   str items  → minimal op dict
    * singleton-object fields (inference, …): list     → {"facts": [...]}

    Applied recursively so nested objects are also fixed.
    """
    if isinstance(data, dict):
        copied: dict[str, Any] = {}
        for k, v in data.items():
            if k in _STR_LIST_FIELDS and isinstance(v, list):
                copied[k] = [
                    _item_to_str(item)
                    for item in v
                    if item is not None
                ]
            elif k in _OPERATION_LIST_FIELDS and isinstance(v, list):
                new_ops: list[Any] = []
                for item in v:
                    if isinstance(item, str):
                        new_ops.append(_str_to_op_dict(item))
                    elif item is not None:
                        new_ops.append(_sanitize_spec_payload_dict(item, k))
                copied[k] = new_ops
            elif k == "nodes" and isinstance(v, list):
                new_nodes: list[Any] = []
                for item in v:
                    if isinstance(item, str):
                        new_nodes.append(_str_to_node_dict(item))
                    elif item is not None:
                        new_nodes.append(_sanitize_spec_payload_dict(item, k))
                copied[k] = new_nodes
            elif k == "edges" and isinstance(v, list):
                new_edges: list[Any] = []
                for item in v:
                    if isinstance(item, str):
                        new_edges.append(_str_to_edge_dict(item))
                    elif item is not None:
                        new_edges.append(_sanitize_spec_payload_dict(item, k))
                copied[k] = new_edges
            elif k == "forward_pass" and isinstance(v, list):
                new_fps: list[Any] = []
                for idx, item in enumerate(v, start=1):
                    if isinstance(item, str):
                        new_fps.append(_str_to_forward_pass_step_dict(item, idx))
                    elif item is not None:
                        new_fps.append(_sanitize_spec_payload_dict(item, k))
                copied[k] = new_fps
            elif k == "tensor_flow" and isinstance(v, list):
                copied[k] = " -> ".join(str(item) for item in v if item is not None)
            elif k == "equations" and isinstance(v, list):
                new_eqs: list[Any] = []
                for item in v:
                    if isinstance(item, str):
                        new_eqs.append(_str_to_executable_op_dict(item))
                    elif item is not None:
                        new_eqs.append(_sanitize_spec_payload_dict(item, k))
                copied[k] = new_eqs
            elif k in _SINGLETON_OBJECT_FIELDS and isinstance(v, list):
                # LLM emitted a list for a field that must be a single object
                copied[k] = _list_to_singleton_object(v, k)
            elif k in _SINGLETON_OBJECT_FIELDS and v is None:
                # LLM emitted null for a required object field — use empty default
                copied[k] = {}
            elif k in _FACT_LIST_FIELDS and isinstance(v, dict):
                # LLM emitted a dict instead of a list of facts
                copied[k] = _sanitize_spec_payload_dict(_dict_to_fact_list(v), k)
            elif k in _COMPONENT_LIST_FIELDS and isinstance(v, dict):
                # LLM emitted a dict instead of a list of components
                copied[k] = _sanitize_spec_payload_dict(_dict_to_component_list(v), k)
            else:
                copied[k] = _sanitize_spec_payload_dict(v, k)
        return copied
    elif isinstance(data, list):
        return [
            _sanitize_spec_payload_dict(item, _parent_key)
            for item in data
        ]
    return data


def validate_spec_payload(
    data: Any,
    context: str,
) -> ImplementationSpec:
    """
    Validate an LLM payload against the Pydantic schema with a useful
    error message.
    """
    data = unwrap_spec_payload(data)
    data = _sanitize_spec_payload_dict(data)

    try:
        return ImplementationSpec.model_validate(
            data
        )

    except ValidationError as exc:
        raise ValueError(
            f"{context} produced JSON that does not match "
            f"ImplementationSpec:\n{exc}"
        ) from exc


# ---------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------

async def analyze_chunk(
    pages: list[dict[str, Any]],
    schema: str,
    chunk_index: int,
    total_chunks: int,
) -> ImplementationSpec:
    paper_text = build_numbered_text(
        pages
    )

    page_numbers = [
        page["page"]
        for page in pages
    ]

    user_prompt = f"""
Analyze the following research-paper chunk.

Chunk:
{chunk_index} of {total_chunks}

Pages:
{page_numbers}

This pipeline uses overlapping chunks.
Therefore some pages may also appear in neighboring chunks.

Your task:
Extract implementation-relevant evidence from THIS chunk.

Important chunk rule:
If this chunk does not mention a detail, omit it.
Do NOT mark a detail UNKNOWN merely because it is absent here.

Architecture rule — Preserve Tensor Flow:
The tensor flow IS the architecture.
Extract exact directed tensor flow (e.g. Image -> PatchEmbedding -> Tokens -> Transformer -> Logits).
Do NOT merely list static components ("Uses PatchEmbedding", "Uses Transformer").
When the chunk explicitly describes connected operations, preserve the relationship.

Example:
If the paper says:
    A has shape [N, D]
    B has shape [N, 4D]
    concatenate A and B to obtain [N, 5D]
    project [N, 5D] to [N, D]

Do not return only four disconnected shape facts.
Also extract the concatenation and projection relationship.

Equation extraction rule — Executable Operations:
Extract equations as deterministic executable operations rather than descriptive names.
Instead of:
    "Scaled Dot Product Attention"
Extract:
    {{"operation": "attention", "formula": "softmax(QK^T/sqrt(dk))V"}}

Numeric evidence rule:
Do not infer hyperparameters from nearby constants.

Example:
    lr = blr * global_batch / 256
does not mean:
    batch_size = 256

SCHEMA:

{schema}

PAPER CHUNK:

{paper_text}
"""

    print(
        "[PaperUnderstanding] "
        f"Sending extraction request "
        f"{chunk_index}/{total_chunks} "
        f"for pages {page_numbers}...",
        flush=True,
    )

    content = await stream_chat_with_retries(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=EXTRACTION_TEMPERATURE,
        max_retries=MAX_RETRIES,
        base_delay=BASE_RETRY_DELAY,
        log_prefix=(
            "[PaperUnderstanding] "
            f"[Extract {chunk_index}/{total_chunks}]"
        ),
    )

    data = extract_json_payload(
        content
    )

    spec = validate_spec_payload(
        data,
        context=(
            f"Chunk {chunk_index}/{total_chunks}"
        ),
    )

    print(
        "[PaperUnderstanding] "
        f"Finished extraction "
        f"{chunk_index}/{total_chunks}: "
        f"{len(spec.model_components)} components, "
        f"{len(spec.preprocessing)} preprocessing facts, "
        f"{len(spec.training)} training facts.",
        flush=True,
    )

    return spec


# ---------------------------------------------------------------------
# Vision extraction (Visual figures / Architecture diagrams)
# ---------------------------------------------------------------------

async def analyze_visual_assets(
    pages: list[dict[str, Any]],
    schema: str,
    enable_vision: bool = ENABLE_VISION_EXTRACTION,
    batch_size: int = 1,  # API limit: at most 1 image per request
) -> list[ImplementationSpec]:
    """
    Extract implementation evidence directly from architecture figures,
    block diagrams, and visual assets across extracted PDF pages.
    """
    if not enable_vision:
        return []

    visual_assets: list[dict[str, Any]] = []
    for page in pages:
        for img in page.get("images", []):
            visual_assets.append(img)

    if not visual_assets:
        print(
            "[PaperUnderstanding] No visual figure assets detected for vision extraction.",
            flush=True,
        )
        return []

    print(
        f"[PaperUnderstanding] Starting vision extraction across {len(visual_assets)} visual architecture figure(s)...",
        flush=True,
    )

    vision_specs: list[ImplementationSpec] = []

    # Process visual assets in batches
    for batch_index in range(0, len(visual_assets), batch_size):
        batch = visual_assets[batch_index : batch_index + batch_size]
        batch_num = (batch_index // batch_size) + 1
        total_batches = (len(visual_assets) + batch_size - 1) // batch_size

        asset_ids = [asset["image_id"] for asset in batch]
        asset_descriptions = []
        for asset in batch:
            caption_hint = asset.get("caption") or "Architecture figure diagram"
            asset_descriptions.append(
                f"- [ASSET {asset['image_id']}] (page {asset['page']}, {asset['width']}x{asset['height']}): {caption_hint}"
            )

        context_summary = "\n".join(asset_descriptions)

        prompt_text = f"""
Analyze the following research-paper visual architecture figure(s)/diagram(s).

Visual assets in batch {batch_num} of {total_batches}:
{context_summary}

Your task:
Extract precise architectural and implementation evidence depicted in these visual figures and diagrams.

Focus on:
1. Model components and block structure shown in the diagrams
2. Forward-pass operations and connections (skip connections, residual links, concatenation, projections)
3. PRESERVE TENSOR FLOW: The tensor flow IS the architecture. Extract exact directed flow chain (Image -> PatchEmbedding -> Tokens -> Transformer -> Logits) rather than disconnected static components
4. Tensor dimensions, channel depths, or patch shapes annotated on the figures
5. Layer sequence depicted by block flow

SCHEMA:

{schema}
"""

        print(
            f"[PaperUnderstanding] [Vision {batch_num}/{total_batches}] Analyzing visual assets {asset_ids}...",
            flush=True,
        )

        # Compact JSON template shown to vision model (avoids schema overload)
        vision_json_template = """{
  "paper_title": "...",
  "task": "...",
  "model_components": [
    {
      "name": "ComponentName",
      "component_type": "LAYER|BLOCK|MODULE|ENCODER|DECODER|HEAD|BACKBONE|OTHER",
      "facts": [
        {"name": "fact_name", "value": "value", "status": "PAPER_REPORTED", "confidence": 0.9,
         "evidence": [{"page": 1, "quote": "visible in figure"}]}
      ]
    }
  ],
  "architecture": {
    "inputs": ["input description"],
    "outputs": ["output description"],
    "operations": [
      {"id": "op_id", "name": "Operation Name", "operation_type": "CONV|LINEAR|NORM|ACTIVATION|POOL|ATTENTION|OTHER|UNKNOWN",
       "inputs": ["tensor_in"], "outputs": ["tensor_out"], "parameters": {}}
    ]
  },
  "preprocessing": [],
  "training": [],
  "unknowns": []
}"""

        prompt_text = f"""You are analyzing a research-paper architecture figure or diagram (image attached).

Figure context:
{context_summary}

Extract implementation evidence VISIBLE in the figure (TREAT FIGURE AS PRIMARY TOPOLOGY SOURCE):
- Model components and layer blocks (name, type, channel dims if annotated)
- Sequential forward-pass flow shown by arrows
- Skip/residual connections, branching pathways, concatenations, projections
- Any annotated tensor shapes or hyperparameters

Return ONLY valid JSON matching this exact template (fill real values, omit empty lists):

```json
{vision_json_template}
```

CRITICAL: Output ONLY the JSON block above. No explanations, no prose.
"""

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text}
        ]
        for asset in batch:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{asset['base64']}"
                },
            })

        try:
            content = await stream_chat_with_retries(
                model=VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": VISION_EXTRACTION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=EXTRACTION_TEMPERATURE,
                max_retries=2,
                base_delay=BASE_RETRY_DELAY,
                log_prefix=f"[PaperUnderstanding] [Vision {batch_num}/{total_batches}]",
            )
        except Exception as exc:
            print(
                f"[PaperUnderstanding] [Vision {batch_num}/{total_batches}] Skipping (vision model unavailable: {exc}).",
                flush=True,
            )
            continue

        try:
            data = extract_json_payload(content)
            spec = validate_spec_payload(
                data,
                context=f"Vision assets {asset_ids}",
            )
            vision_specs.append(spec)
            print(
                f"[PaperUnderstanding] [Vision {batch_num}/{total_batches}] Extracted {len(spec.model_components)} components from visual assets.",
                flush=True,
            )
        except Exception:
            # Silently skip non-JSON or non-architecture visual assets without warning
            pass

    return vision_specs


# ---------------------------------------------------------------------
# Conservative local normalization
# ---------------------------------------------------------------------

_CANONICAL_FACT_NAMES = {
    "optimizer": "optimizer",
    "optimiser": "optimizer",
    "number of epochs": "epochs",
    "num epochs": "epochs",
    "epoch count": "epochs",
    "epochs": "epochs",
    "batch size": "batch_size",
    "global batch size": "global_batch_size",
    "learning rate": "learning_rate",
    "base learning rate": "base_learning_rate",
    "effective learning rate": "effective_learning_rate",
    "learning rate schedule": "learning_rate_schedule",
    "learning-rate schedule": "learning_rate_schedule",
    "learning rate scaling rule": "learning_rate_scaling_rule",
    "learning-rate scaling rule": "learning_rate_scaling_rule",
    "input shape": "input_shape",
    "output shape": "output_shape",
    "training resolution": "training_resolution",
    "evaluation resolution": "evaluation_resolution",
    "patch size": "patch_size",
    "prediction target": "prediction_target",
    "prediction parameterization": "prediction_parameterization",
    "training loss": "training_loss",
    "loss function": "loss_function",
    "noise schedule": "noise_schedule",
    "number of transformer blocks": "transformer_block_count",
}


def normalize_text_key(
    value: str | None
) -> str:
    if not value:
        return ""

    value = value.strip().lower()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)

    return value


def canonical_fact_name(
    name: str
) -> str:
    normalized = normalize_text_key(name)

    return _CANONICAL_FACT_NAMES.get(
        normalized,
        normalized.replace(" ", "_"),
    )


_COMPONENT_TYPE_ALIASES = {
    "diffusion transformer": "diffusion_transformer",
    "pixel space diffusion model": "diffusion_transformer",
    "pixel-space diffusion model": "diffusion_transformer",
    "diffusion model": "diffusion_model",
    "transformer": "transformer",
    "backbone": "backbone",
    "linear layer": "linear_layer",
}


def canonical_component_type(
    component_type: str | None
) -> str | None:
    if component_type is None:
        return None

    normalized = normalize_text_key(
        component_type
    )

    return _COMPONENT_TYPE_ALIASES.get(
        normalized,
        normalized.replace(" ", "_"),
    )


def canonical_component_name(
    name: str
) -> str:
    normalized = normalize_text_key(name)

    # Keep the semantic identity while normalizing superficial variation.
    return normalized


def evidence_key(
    evidence: Evidence
) -> tuple[Any, ...]:
    return (
        evidence.page,
        normalize_text_key(evidence.section),
        normalize_text_key(evidence.quote),
    )


def fact_key(
    fact: ImplementationFact
) -> tuple[Any, ...]:
    value_key = json.dumps(
        fact.value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return (
        canonical_fact_name(fact.name),
        value_key,
        fact.status,
    )


def merge_evidence_lists(
    existing: list[Evidence],
    incoming: list[Evidence],
) -> list[Evidence]:
    merged: OrderedDict[
        tuple[Any, ...],
        Evidence
    ] = OrderedDict()

    for item in existing + incoming:
        merged[evidence_key(item)] = item

    return list(merged.values())


def merge_fact_lists(
    existing: list[ImplementationFact],
    incoming: list[ImplementationFact],
) -> list[ImplementationFact]:
    """
    Conservative local merge.

    Important:
    Facts with different values are NOT collapsed.
    The global synthesis pass resolves semantic conflicts.
    """
    merged: OrderedDict[
        tuple[Any, ...],
        ImplementationFact
    ] = OrderedDict()

    for fact in existing + incoming:
        key = fact_key(fact)

        if key not in merged:
            merged[key] = ImplementationFact(
                name=fact.name,
                value=fact.value,
                status=fact.status,
                confidence=fact.confidence,
                evidence=list(fact.evidence),
                notes=fact.notes,
            )
            continue

        current = merged[key]

        current.evidence = merge_evidence_lists(
            current.evidence,
            fact.evidence,
        )

        current.confidence = max(
            current.confidence,
            fact.confidence,
        )

        if (
            not current.notes
            and fact.notes
        ):
            current.notes = fact.notes

    return list(merged.values())


def component_key(
    component: ModelComponent
) -> tuple[str, str | None]:
    return (
        canonical_component_name(
            component.name
        ),
        canonical_component_type(
            component.component_type
        ),
    )


def merge_components(
    existing: list[ModelComponent],
    incoming: list[ModelComponent],
) -> list[ModelComponent]:
    """
    Conservative merge before LLM synthesis.

    This removes exact/superficial duplicates but intentionally does not
    pretend Python can solve all semantic component identity problems.
    """
    merged: OrderedDict[
        tuple[str, str | None],
        ModelComponent
    ] = OrderedDict()

    for component in existing + incoming:
        key = component_key(
            component
        )

        if key not in merged:
            merged[key] = ModelComponent(
                name=component.name,
                component_type=component.component_type,
                facts=list(component.facts),
            )
            continue

        merged[key].facts = merge_fact_lists(
            merged[key].facts,
            component.facts,
        )

    return list(merged.values())


def merge_specs(
    specs: list[ImplementationSpec]
) -> ImplementationSpec:
    """Backward-compatible alias for preliminary_merge_specs."""
    return preliminary_merge_specs(specs)


def merge_architecture_graphs(graphs: list[ArchitectureGraph]) -> ArchitectureGraph:
    """
    Merge architecture graphs, treating figures as the PRIMARY TOPOLOGY SOURCE.
    Figures often contain arrows, branches, skips, and residuals that text never mentions.
    If any graph has primary_topology_source == 'FIGURE', its topology takes primary precedence.
    """
    if not graphs:
        return ArchitectureGraph()

    primary_fig = next((g for g in graphs if getattr(g, "primary_topology_source", "TEXT") == "FIGURE"), None)
    base = primary_fig if primary_fig else graphs[0]

    merged = ArchitectureGraph(
        nodes=list(base.nodes),
        edges=list(base.edges),
        inputs=list(base.inputs),
        outputs=list(base.outputs),
        tensors=list(base.tensors),
        operations=list(base.operations),
        connections=list(base.connections),
        forward_pass=list(base.forward_pass),
        tensor_flow=base.tensor_flow,
        primary_topology_source=base.primary_topology_source,
        branches=list(getattr(base, "branches", [])),
        skips=list(getattr(base, "skips", [])),
        residuals=list(getattr(base, "residuals", [])),
    )

    for g in graphs:
        if g is base:
            continue
        if not merged.tensor_flow and g.tensor_flow:
            merged.tensor_flow = g.tensor_flow
        for b in getattr(g, "branches", []):
            if b not in merged.branches:
                merged.branches.append(b)
        for s in getattr(g, "skips", []):
            if s not in merged.skips:
                merged.skips.append(s)
        for r in getattr(g, "residuals", []):
            if r not in merged.residuals:
                merged.residuals.append(r)

    return merged


def preliminary_merge_specs(
    specs: list[ImplementationSpec]
) -> ImplementationSpec:
    """
    Perform only conservative aggregation.

    This is NOT the final architecture merge.
    Global synthesis happens afterward.
    """
    merged = ImplementationSpec()

    for spec in specs:
        if (
            not merged.paper_title
            and spec.paper_title
        ):
            merged.paper_title = spec.paper_title

        if (
            not merged.task
            and spec.task
        ):
            merged.task = spec.task

        merged.model_components = merge_components(
            merged.model_components,
            spec.model_components,
        )

        merged.preprocessing = merge_fact_lists(
            merged.preprocessing,
            spec.preprocessing,
        )

        merged.training = merge_fact_lists(
            merged.training,
            spec.training,
        )

        merged.unknowns = merge_fact_lists(
            merged.unknowns,
            spec.unknowns,
        )

    arch_graphs = [s.architecture_graph for s in specs if s.architecture_graph]
    if arch_graphs:
        merged.architecture_graph = merge_architecture_graphs(arch_graphs)
        merged.architecture = merged.architecture_graph

    return merged


# ---------------------------------------------------------------------
# Global synthesis
# ---------------------------------------------------------------------

def serialize_chunk_specs(
    specs: list[ImplementationSpec]
) -> str:
    payload = []

    for index, spec in enumerate(
        specs,
        start=1
    ):
        payload.append({
            "chunk_index": index,
            "spec": spec.model_dump(
                mode="json"
            ),
        })

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )


async def synthesize_global_spec(
    chunk_specs: list[ImplementationSpec],
    preliminary_spec: ImplementationSpec,
    schema: str,
) -> ImplementationSpec:
    """
    Reconstruct a coherent global spec from all chunk evidence.
    """
    chunk_payload = serialize_chunk_specs(
        chunk_specs
    )

    preliminary_payload = (
        preliminary_spec.model_dump_json(
            indent=2
        )
    )

    user_prompt = f"""
Reconstruct one globally coherent implementation specification.

You are given:

A. the required schema
B. all overlapping chunk-level extraction results
C. a conservative preliminary merge

Important:
The preliminary merge is NOT authoritative.
It may contain:
- duplicate components
- duplicate facts
- conflicting facts
- disconnected architecture fragments

Your job is to synthesize, not merely copy.

SCHEMA:

{schema}

ALL CHUNK EXTRACTIONS:

{chunk_payload}

PRELIMINARY MERGE:

{preliminary_payload}

Return one corrected, globally coherent JSON specification.
"""

    print(
        "[PaperUnderstanding] "
        "Starting global synthesis pass...",
        flush=True,
    )

    content = await stream_chat_with_retries(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=SYNTHESIS_TEMPERATURE,
        max_retries=MAX_RETRIES,
        base_delay=BASE_RETRY_DELAY,
        log_prefix=(
            "[PaperUnderstanding] "
            "[GlobalSynthesis]"
        ),
    )

    data = extract_json_payload(
        content
    )

    spec = validate_spec_payload(
        data,
        context="Global synthesis",
    )

    print(
        "[PaperUnderstanding] "
        "Global synthesis completed: "
        f"{len(spec.model_components)} components.",
        flush=True,
    )

    return spec


# ---------------------------------------------------------------------
# Full-paper verification
# ---------------------------------------------------------------------

def build_verification_text(
    pages: list[dict[str, Any]]
) -> str:
    """
    Build the full numbered paper text used by the verifier.

    For a 16-page paper this is usually manageable. If you later process
    very long papers, replace this with evidence-targeted retrieval.
    """
    return build_numbered_text(
        pages
    )


async def verify_global_spec(
    spec: ImplementationSpec,
    pages: list[dict[str, Any]],
    schema: str,
) -> ImplementationSpec:
    """
    Verify synthesized claims against the full numbered paper text.
    """
    paper_text = build_verification_text(
        pages
    )

    spec_json = spec.model_dump_json(
        indent=2
    )

    user_prompt = f"""
Verify and correct the following synthesized implementation specification
against the supplied numbered paper text.

Do not merely proofread JSON.
Check semantic support for every claim.

Pay special attention to:
- numeric hyperparameters
- batch size
- learning rate
- base learning rate
- stage-specific resolutions
- input/output shapes
- architecture connections
- prediction target
- optimization loss
- model variants
- evidence page correctness

SCHEMA:

{schema}

SYNTHESIZED SPEC:

{spec_json}

NUMBERED PAPER TEXT:

{paper_text}

Return the corrected full JSON specification only.
"""

    print(
        "[PaperUnderstanding] "
        "Starting claim-evidence verification pass...",
        flush=True,
    )

    content = await stream_chat_with_retries(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": VERIFICATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=VERIFICATION_TEMPERATURE,
        max_retries=MAX_RETRIES,
        base_delay=BASE_RETRY_DELAY,
        log_prefix=(
            "[PaperUnderstanding] "
            "[Verification]"
        ),
    )

    data = extract_json_payload(
        content
    )

    verified_spec = validate_spec_payload(
        data,
        context="Verification pass",
    )

    print(
        "[PaperUnderstanding] "
        "Claim-evidence verification completed.",
        flush=True,
    )

    return verified_spec


# ---------------------------------------------------------------------
# Deterministic post-verification safeguards
# ---------------------------------------------------------------------

def _contains_reference_batch_formula(
    fact: ImplementationFact
) -> bool:
    """
    Detect the specific dangerous pattern that previously produced the
    false batch_size=256 extraction.
    """
    evidence_text = " ".join(
        item.quote or ""
        for item in fact.evidence
    ).lower()

    compact = re.sub(
        r"\s+",
        " ",
        evidence_text
    )

    has_global_batch = (
        "global batch" in compact
        or "global_batch" in compact
    )

    has_div_256 = bool(
        re.search(
            r"(?:/|÷)\s*256",
            compact
        )
    )

    return (
        has_global_batch
        and has_div_256
    )


def _has_direct_batch_assertion(
    fact: ImplementationFact
) -> bool:
    """
    Look for direct evidence such as:
        global batch size is 1024
        batch size of 256

    This is intentionally conservative.
    """
    patterns = [
        r"\bbatch size\s*(?:is|=|of|:)\s*\d+",
        r"\bglobal batch size\s*(?:is|=|of|:)\s*\d+",
        r"\bglobal batch\s*(?:is|=|of|:)\s*\d+",
    ]

    for item in fact.evidence:
        quote = (
            item.quote
            or ""
        ).lower()

        for pattern in patterns:
            if re.search(pattern, quote):
                return True

    return False


def apply_deterministic_safeguards(
    spec: ImplementationSpec
) -> ImplementationSpec:
    """
    Apply narrow deterministic safeguards after LLM verification.

    This does not replace semantic verification.
    It catches known high-risk extraction patterns.
    """
    cleaned_training: list[ImplementationFact] = []

    for fact in spec.training:
        canonical_name = canonical_fact_name(
            fact.name
        )

        if canonical_name in {
            "batch_size",
            "global_batch_size",
        }:
            if (
                _contains_reference_batch_formula(fact)
                and not _has_direct_batch_assertion(fact)
            ):
                cleaned_training.append(
                    ImplementationFact(
                        name=fact.name,
                        value=None,
                        status="UNKNOWN",
                        confidence=0.0,
                        evidence=list(fact.evidence),
                        notes=(
                            "Rejected unsupported batch-size value: "
                            "the cited evidence contains a learning-rate "
                            "scaling formula with reference denominator "
                            "256 but does not directly state the batch size."
                        ),
                    )
                )
                continue

        cleaned_training.append(
            fact
        )

    spec.training = merge_fact_lists(
        [],
        cleaned_training,
    )

    return spec


# ---------------------------------------------------------------------
# Main paper analysis pipeline
# ---------------------------------------------------------------------

async def analyze_paper(
    pdf_path: str,
    enable_vision: bool = ENABLE_VISION_EXTRACTION,
) -> ImplementationSpec:
    print(
        "[PaperUnderstanding] "
        f"Analyzing PDF: {pdf_path} (vision={enable_vision})",
        flush=True,
    )

    # -------------------------------------------------------------
    # 1. Structured extraction (Text + Visual Assets).
    # -------------------------------------------------------------
    pages = extract_pdf_pages(
        pdf_path,
        extract_images=enable_vision,
    )

    if not pages:
        raise ValueError(
            "No pages were extracted from the PDF"
        )

    schema = build_schema_prompt()

    # -------------------------------------------------------------
    # 2. Overlapping chunks.
    # -------------------------------------------------------------
    page_chunks = split_pages_into_chunks(
        pages,
        pages_per_chunk=PAGES_PER_CHUNK,
        overlap_pages=OVERLAP_PAGES,
    )

    print(
        "[PaperUnderstanding] "
        f"Extracted {len(pages)} pages.",
        flush=True,
    )

    total_visual_assets = sum(len(page.get("images", [])) for page in pages)
    print(
        "[PaperUnderstanding] "
        f"Extracted {total_visual_assets} visual figure asset(s) across pages.",
        flush=True,
    )

    print(
        "[PaperUnderstanding] "
        f"Created {len(page_chunks)} overlapping chunks "
        f"(size={PAGES_PER_CHUNK}, "
        f"overlap={OVERLAP_PAGES}).",
        flush=True,
    )

    for index, chunk in enumerate(
        page_chunks,
        start=1
    ):
        page_numbers = [
            page["page"]
            for page in chunk
        ]

        print(
            "[PaperUnderstanding] "
            f"Chunk {index}: pages {page_numbers}",
            flush=True,
        )

    # -------------------------------------------------------------
    # 3. Chunk-level evidence extraction.
    #
    # Sequential by default for API stability.
    # -------------------------------------------------------------
    chunk_specs: list[ImplementationSpec] = []

    for index, chunk in enumerate(
        page_chunks,
        start=1
    ):
        spec = await analyze_chunk(
            pages=chunk,
            schema=schema,
            chunk_index=index,
            total_chunks=len(page_chunks),
        )

        chunk_specs.append(
            spec
        )

    print(
        "[PaperUnderstanding] "
        "All text chunk extractions completed.",
        flush=True,
    )

    # -------------------------------------------------------------
    # 3b. Visual architecture evidence extraction (Vision pass).
    # -------------------------------------------------------------
    if enable_vision and total_visual_assets > 0:
        vision_specs = await analyze_visual_assets(
            pages=pages,
            schema=schema,
            enable_vision=enable_vision,
        )
        if vision_specs:
            print(
                "[PaperUnderstanding] "
                f"Aggregating {len(vision_specs)} visual architecture spec(s) with text chunk specs.",
                flush=True,
            )
            chunk_specs.extend(vision_specs)

    # -------------------------------------------------------------
    # 4. Conservative preliminary merge.
    # -------------------------------------------------------------
    preliminary_spec = preliminary_merge_specs(
        chunk_specs
    )

    print(
        "[PaperUnderstanding] "
        "Preliminary evidence aggregation completed.",
        flush=True,
    )

    # -------------------------------------------------------------
    # 5. Global synthesis.
    # -------------------------------------------------------------
    synthesized_spec = await synthesize_global_spec(
        chunk_specs=chunk_specs,
        preliminary_spec=preliminary_spec,
        schema=schema,
    )

    # -------------------------------------------------------------
    # 6. Full-paper claim-evidence verification.
    # -------------------------------------------------------------
    verified_spec = await verify_global_spec(
        spec=synthesized_spec,
        pages=pages,
        schema=schema,
    )

    # -------------------------------------------------------------
    # 7. Narrow deterministic safeguards.
    # -------------------------------------------------------------
    final_spec = apply_deterministic_safeguards(
        verified_spec
    )

    print(
        "[PaperUnderstanding] "
        "Final safeguards completed.",
        flush=True,
    )

    return final_spec


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def resolve_pdf_path(
    argument: str | None
) -> Path:
    """
    Resolve:
    - explicit filesystem path
    - filename under ./papers
    - default PointDiT paper
    """
    module_dir = Path(__file__).resolve().parent

    if argument:
        candidate = Path(argument)

        if candidate.exists():
            return candidate.resolve()

        paper_candidate = (
            module_dir
            / "papers"
            / argument
        )

        if paper_candidate.exists():
            return paper_candidate.resolve()

        raise FileNotFoundError(
            f'Could not find PDF "{argument}" '
            f"as a direct path or under "
            f'"{module_dir / "papers"}".'
        )

    default_path = (
        module_dir
        / "papers"
        / "2607.02515v1.pdf"
    )

    if not default_path.exists():
        raise FileNotFoundError(
            "No PDF argument was supplied and the default "
            f'paper does not exist at "{default_path}".'
        )

    return default_path.resolve()


async def main() -> None:
    argument = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    pdf_path = resolve_pdf_path(
        argument
    )

    spec = await analyze_paper(
        str(pdf_path)
    )

    output_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / OUTPUT_FILENAME
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        spec.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    print(
        "\nImplementation specification created",
        flush=True,
    )

    print(
        f"Saved to: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())