from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

EvidenceStatus = Literal[
    "PAPER_REPORTED",
    "INFERRED",
    "ASSUMED",
    "UNKNOWN",
    "AMBIGUOUS",
    "REFERENCED_ELSEWHERE",
    "REGISTRY_CANONICAL",
    "LITERATURE_GROUNDED",
    "DOMAIN_HEURISTIC",
]


EvidenceType = Literal[
    "DIRECT_QUOTE",
    "EQUATION",
    "TABLE",
    "FIGURE",
    "CAPTION",
    "APPENDIX",
    "DERIVED",
    "UNKNOWN",
]


SupportStrength = Literal[
    "DIRECT",
    "STRONG",
    "INDIRECT",
    "WEAK",
    "NONE",
    "UNKNOWN",
]


ComponentRole = Literal[
    "MODEL",
    "BACKBONE",
    "ENCODER",
    "DECODER",
    "TRANSFORMER",
    "ATTENTION",
    "MLP",
    "EMBEDDING",
    "PROJECTION",
    "HEAD",
    "NORMALIZATION",
    "ACTIVATION",
    "FUSION",
    "PATCHIFY",
    "UNPATCHIFY",
    "PREPROCESSOR",
    "LOSS",
    "CONDITIONER",
    "SAMPLER",
    "SOLVER",
    "OTHER",
    "UNKNOWN",
]


OperationType = Literal[
    "INPUT",
    "OUTPUT",
    "PATCHIFY",
    "UNPATCHIFY",
    "FLATTEN",
    "RESHAPE",
    "PERMUTE",
    "CONCAT",
    "ADD",
    "MULTIPLY",
    "LINEAR",
    "CONVOLUTION",
    "POOLING",
    "SELF_ATTENTION",
    "CROSS_ATTENTION",
    "MLP",
    "NORMALIZATION",
    "ACTIVATION",
    "ENCODE",
    "DECODE",
    "CONDITION",
    "PROJECT",
    "SAMPLE",
    "ODE_STEP",
    "OTHER",
    "UNKNOWN",
]


ConnectionType = Literal[
    "SEQUENTIAL",
    "RESIDUAL",
    "CONCATENATION",
    "ADDITION",
    "CONDITIONING",
    "SKIP",
    "BRANCH",
    "MERGE",
    "OTHER",
    "UNKNOWN",
]


FactCategory = Literal[
    "ARCHITECTURE",
    "SHAPE",
    "DIMENSION",
    "HYPERPARAMETER",
    "PREPROCESSING",
    "TRAINING",
    "LOSS",
    "OPTIMIZER",
    "SCHEDULER",
    "INFERENCE",
    "DATASET",
    "AUGMENTATION",
    "VARIANT",
    "REFERENCE",
    "OTHER",
    "UNKNOWN",
]


ValueType = Literal[
    "STRING",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "SHAPE",
    "RANGE",
    "LIST",
    "MAPPING",
    "FORMULA",
    "NULL",
    "UNKNOWN",
]


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

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


VALID_EVIDENCE_TYPES = {
    "DIRECT_QUOTE",
    "EQUATION",
    "TABLE",
    "FIGURE",
    "CAPTION",
    "APPENDIX",
    "DERIVED",
    "UNKNOWN",
}


VALID_SUPPORT_STRENGTHS = {
    "DIRECT",
    "STRONG",
    "INDIRECT",
    "WEAK",
    "NONE",
    "UNKNOWN",
}


VALID_COMPONENT_ROLES = {
    "MODEL",
    "BACKBONE",
    "ENCODER",
    "DECODER",
    "TRANSFORMER",
    "ATTENTION",
    "MLP",
    "EMBEDDING",
    "PROJECTION",
    "HEAD",
    "NORMALIZATION",
    "ACTIVATION",
    "FUSION",
    "PATCHIFY",
    "UNPATCHIFY",
    "PREPROCESSOR",
    "LOSS",
    "CONDITIONER",
    "SAMPLER",
    "SOLVER",
    "OTHER",
    "UNKNOWN",
}


VALID_OPERATION_TYPES = {
    "INPUT",
    "OUTPUT",
    "PATCHIFY",
    "UNPATCHIFY",
    "FLATTEN",
    "RESHAPE",
    "PERMUTE",
    "CONCAT",
    "ADD",
    "MULTIPLY",
    "LINEAR",
    "CONVOLUTION",
    "POOLING",
    "SELF_ATTENTION",
    "CROSS_ATTENTION",
    "MLP",
    "NORMALIZATION",
    "ACTIVATION",
    "ENCODE",
    "DECODE",
    "CONDITION",
    "PROJECT",
    "SAMPLE",
    "ODE_STEP",
    "OTHER",
    "UNKNOWN",
}


VALID_CONNECTION_TYPES = {
    "SEQUENTIAL",
    "RESIDUAL",
    "CONCATENATION",
    "ADDITION",
    "CONDITIONING",
    "SKIP",
    "BRANCH",
    "MERGE",
    "OTHER",
    "UNKNOWN",
}


VALID_FACT_CATEGORIES = {
    "ARCHITECTURE",
    "SHAPE",
    "DIMENSION",
    "HYPERPARAMETER",
    "PREPROCESSING",
    "TRAINING",
    "LOSS",
    "OPTIMIZER",
    "SCHEDULER",
    "INFERENCE",
    "DATASET",
    "AUGMENTATION",
    "VARIANT",
    "REFERENCE",
    "OTHER",
    "UNKNOWN",
}


VALID_VALUE_TYPES = {
    "STRING",
    "INTEGER",
    "FLOAT",
    "BOOLEAN",
    "SHAPE",
    "RANGE",
    "LIST",
    "MAPPING",
    "FORMULA",
    "NULL",
    "UNKNOWN",
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _coerce_str_list(value: Any) -> list[str]:
    """
    Coerce flexible LLM outputs (including dicts or scalars) into list[str].
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        name = value.get("name") or value.get("id")
        dims = value.get("dimensions") or value.get("shape")
        if name:
            if dims:
                return [f"{name} {dims}"]
            return [str(name)]
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("id")
                dims = item.get("dimensions") or item.get("shape")
                if name:
                    if dims:
                        result.append(f"{name} {dims}")
                    else:
                        result.append(str(name))
                else:
                    result.append(str(item))
            else:
                result.append(str(item))
        return result
    return [str(value)]


def _normalize_space(
    value: str
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value.strip(),
    )


def _normalize_identifier(
    value: str
) -> str:
    value = _normalize_space(value)
    value = value.lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(
        r"[^a-z0-9_]+",
        "",
        value,
    )
    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def _is_finite_number(
    value: Any
) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _coerce_enum(
    value: Any,
    valid_values: set[str],
    default: str,
) -> str:
    if not isinstance(value, str):
        return default

    normalized = (
        value
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in valid_values:
        return normalized

    return default


def _infer_value_type(
    value: Any
) -> str:
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "BOOLEAN"

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return "INTEGER"

    if isinstance(value, float):
        return "FLOAT"

    if isinstance(value, list):
        return "LIST"

    if isinstance(value, dict):
        return "MAPPING"

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return "STRING"

        if re.search(
            r"\b(?:R\^|ℝ|\[.*\]|×|x)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            return "SHAPE"

        if any(
            token in stripped
            for token in [
                "=",
                "->",
                "→",
                "sigmoid(",
                "sqrt(",
                "log(",
            ]
        ):
            return "FORMULA"

        return "STRING"

    return "UNKNOWN"


# ---------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------

class SafeBaseModel(BaseModel):
    """
    Shared defensive coercion.

    The LLM may occasionally emit null for list/dict fields.
    Convert those to empty collections before validation.
    """

    @model_validator(mode="before")
    @classmethod
    def coerce_none_collections(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        for key, value in list(copied.items()):
            if (
                value is not None
                or key not in cls.model_fields
            ):
                continue

            field_info = cls.model_fields[key]
            annotation = field_info.annotation

            annotation_text = (
                str(annotation)
                if annotation is not None
                else ""
            )

            if (
                "list" in annotation_text.lower()
                or "List" in annotation_text
            ):
                copied[key] = []

            elif (
                "dict" in annotation_text.lower()
                or "Dict" in annotation_text
            ):
                copied[key] = {}

        return copied


# ---------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------

class Evidence(SafeBaseModel):
    """
    Evidence supporting an extracted implementation claim.
    """

    page: int | None = Field(
        default=None,
        ge=1,
    )

    section: str | None = None

    quote: str | None = None

    evidence_type: EvidenceType = "UNKNOWN"

    support_strength: SupportStrength = "UNKNOWN"

    source_label: str | None = None

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_evidence(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        copied["evidence_type"] = _coerce_enum(
            copied.get(
                "evidence_type",
                "UNKNOWN",
            ),
            VALID_EVIDENCE_TYPES,
            "UNKNOWN",
        )

        copied["support_strength"] = _coerce_enum(
            copied.get(
                "support_strength",
                "UNKNOWN",
            ),
            VALID_SUPPORT_STRENGTHS,
            "UNKNOWN",
        )

        for key in [
            "section",
            "quote",
            "source_label",
            "notes",
        ]:
            value = copied.get(key)

            if isinstance(value, str):
                copied[key] = value.strip() or None

        return copied


# ---------------------------------------------------------------------
# Typed values
# ---------------------------------------------------------------------

class ShapeSpec(SafeBaseModel):
    """
    Structured tensor shape.

    Examples:
        ["H", "W", 3]
        ["N", "D"]
        ["N", "4D"]
    """

    dimensions: list[str | int] = Field(
        default_factory=list
    )

    layout: str | None = None

    semantic: str | None = None

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_shape(
        cls,
        data: Any,
    ) -> Any:
        if isinstance(data, list):
            return {
                "dimensions": data
            }

        if isinstance(data, str):
            cleaned = (
                data
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
            )

            parts = re.split(
                r"\s*[×x,]\s*",
                cleaned,
            )

            dimensions: list[str | int] = []

            for part in parts:
                part = part.strip()

                if not part:
                    continue

                if part.isdigit():
                    dimensions.append(
                        int(part)
                    )
                else:
                    dimensions.append(
                        part
                    )

            return {
                "dimensions": dimensions
            }

        return data


class RangeSpec(SafeBaseModel):
    minimum: int | float | None = None
    maximum: int | float | None = None
    inclusive_minimum: bool = True
    inclusive_maximum: bool = True
    unit: str | None = None


class FormulaSpec(SafeBaseModel):
    expression: str
    variables: dict[str, str] = Field(
        default_factory=dict
    )
    notes: str | None = None


class FactValue(SafeBaseModel):
    """
    Typed wrapper for implementation values.

    `raw` remains for compatibility with arbitrary paper facts,
    but `value_type` makes the semantics explicit.
    """

    value_type: ValueType = "UNKNOWN"

    raw: (
        str
        | int
        | float
        | bool
        | list[Any]
        | dict[str, Any]
        | None
    ) = None

    shape: ShapeSpec | None = None

    range: RangeSpec | None = None

    formula: FormulaSpec | None = None

    unit: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_raw_values(
        cls,
        data: Any,
    ) -> Any:
        # Allow:
        # value = 256
        # value = "AdamW"
        # value = [256, 512]
        if not isinstance(data, dict):
            return {
                "value_type": _infer_value_type(
                    data
                ),
                "raw": data,
            }

        copied = dict(data)

        if "value_type" not in copied:
            copied["value_type"] = (
                _infer_value_type(
                    copied.get("raw")
                )
            )

        copied["value_type"] = _coerce_enum(
            copied.get("value_type"),
            VALID_VALUE_TYPES,
            "UNKNOWN",
        )

        return copied


# ---------------------------------------------------------------------
# Implementation facts
# ---------------------------------------------------------------------

def _infer_fact_required(name: str) -> bool:
    name_lower = (name or "").strip().lower()
    if not name_lower:
        return False
    optional_keywords = {
        "optimizer",
        "scheduler",
        "epoch",
        "batch",
        "dropout",
        "kernel",
        "lr",
        "learning_rate",
        "weight_decay",
        "warmup",
    }
    if any(k in name_lower for k in optional_keywords):
        return False
    required_keywords = {
        "layer",
        "count",
        "topology",
        "dimension",
        "dim",
        "forward",
        "order",
        "shape",
        "tensor",
        "head",
        "channel",
        "architecture",
        "input",
        "output",
    }
    if any(k in name_lower for k in required_keywords):
        return True
    return False


class ImplementationFact(SafeBaseModel):
    """
    One implementation-relevant claim.

    Compatibility note:
    `value` still accepts primitive JSON values because the corrected
    client.py currently emits them. The additional `typed_value` field
    provides stronger semantics without breaking the pipeline.
    """

    name: str

    value: (
        str
        | int
        | float
        | bool
        | list[Any]
        | dict[str, Any]
        | None
    ) = None

    typed_value: FactValue | None = None

    category: FactCategory = "UNKNOWN"

    status: EvidenceStatus = "UNKNOWN"

    required: bool = False

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_fact(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        copied["status"] = _coerce_enum(
            copied.get(
                "status",
                "UNKNOWN",
            ),
            VALID_STATUSES,
            "UNKNOWN",
        )

        copied["category"] = _coerce_enum(
            copied.get(
                "category",
                "UNKNOWN",
            ),
            VALID_FACT_CATEGORIES,
            "UNKNOWN",
        )

        if "required" in copied and copied["required"] is not None:
            val = copied["required"]
            if isinstance(val, str):
                copied["required"] = val.strip().lower() == "true"
            else:
                copied["required"] = bool(val)
        else:
            copied["required"] = _infer_fact_required(copied.get("name", ""))

        confidence = copied.get(
            "confidence"
        )

        if not _is_finite_number(confidence):
            copied["confidence"] = 0.5

        if isinstance(
            copied.get("name"),
            str,
        ):
            copied["name"] = (
                _normalize_space(
                    copied["name"]
                )
            )

        if isinstance(
            copied.get("notes"),
            str,
        ):
            copied["notes"] = (
                copied["notes"].strip()
                or None
            )

        # Automatically create typed semantics from the legacy value.
        if (
            "typed_value" not in copied
            and "value" in copied
        ):
            copied["typed_value"] = {
                "value_type": _infer_value_type(
                    copied.get("value")
                ),
                "raw": copied.get("value"),
            }

        return copied

    @model_validator(mode="after")
    def validate_claim_semantics(
        self,
    ) -> "ImplementationFact":
        """
        Narrow semantic safeguards.

        These do not attempt full natural-language entailment.
        They catch dangerous structural inconsistencies.
        """

        # UNKNOWN should not pretend to have certainty.
        if self.status == "UNKNOWN":
            self.confidence = min(
                self.confidence,
                0.5,
            )

        # PAPER_REPORTED without evidence is suspicious.
        # Keep it valid for compatibility, but cap confidence.
        if (
            self.status == "PAPER_REPORTED"
            and not self.evidence
        ):
            self.confidence = min(
                self.confidence,
                0.6,
            )

        # No evidence support should not carry perfect confidence.
        if self.evidence:
            strengths = {
                item.support_strength
                for item in self.evidence
            }

            if strengths == {"NONE"}:
                self.confidence = min(
                    self.confidence,
                    0.2,
                )

        return self


# ---------------------------------------------------------------------
# Tensor specifications
# ---------------------------------------------------------------------

class TensorSpec(SafeBaseModel):
    """
    A named tensor in the architecture graph.
    """

    id: str

    name: str

    shape: ShapeSpec | None = None

    dtype: str | None = None

    semantic: str | None = None

    producer: str | None = None

    consumers: list[str] = Field(
        default_factory=list
    )

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def sanitize_tensor(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        if not copied.get("id"):
            name = copied.get(
                "name",
                "tensor",
            )

            copied["id"] = (
                _normalize_identifier(
                    str(name)
                )
                or "tensor"
            )

        return copied


# ---------------------------------------------------------------------
# Architecture operations
# ---------------------------------------------------------------------

class ForwardPassStep(SafeBaseModel):
    """
    Explicit forward pass step contract for code generation.

    Example:
        {
            "step": 3,
            "operation": "Encoder",
            "input": "embedded_tokens",
            "output": "memory"
        }
    """

    step: int = 1
    operation: str
    input: str | list[str] = ""
    output: str | list[str] = ""
    consumer_operation: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_forward_pass_step(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)
        if "step" not in copied:
            copied["step"] = copied.get("order", 1) or 1

        if "operation" not in copied:
            copied["operation"] = copied.get("name") or copied.get("id") or "layer"

        if "input" not in copied:
            inps = copied.get("inputs", [])
            if isinstance(inps, list):
                copied["input"] = ", ".join(str(i) for i in inps) if len(inps) > 1 else (str(inps[0]) if inps else "x")
            else:
                copied["input"] = str(inps) if inps else "x"

        if "output" not in copied:
            outs = copied.get("outputs", [])
            if isinstance(outs, list):
                copied["output"] = ", ".join(str(o) for o in outs) if len(outs) > 1 else (str(outs[0]) if outs else "out")
            else:
                copied["output"] = str(outs) if outs else "out"

        if "consumer_operation" not in copied:
            copied["consumer_operation"] = copied.get("consumer") or copied.get("next_operation")

        return copied

    def to_code_line(self) -> str:
        op_name = _normalize_identifier(self.operation) or "layer"
        inp_str = self.input if isinstance(self.input, str) else ", ".join(str(i) for i in self.input)
        out_str = self.output if isinstance(self.output, str) else ", ".join(str(o) for o in self.output)
        return f"{out_str} = self.{op_name}({inp_str})"


class ArchitectureOperation(SafeBaseModel):
    """
    One explicit operation in the forward graph.

    Example:
        Tc [N,4D] + Tz [N,D]
            -> CONCAT
            -> Tin [N,5D]
    """

    id: str

    name: str

    operation_type: OperationType = "UNKNOWN"

    step: int | None = None

    operation: str | None = None

    input: str | list[str] | None = None

    output: str | list[str] | None = None

    inputs: list[str] = Field(
        default_factory=list
    )

    outputs: list[str] = Field(
        default_factory=list
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    repeat_count: int | str | None = None

    order: int | None = Field(
        default=None,
        ge=0,
    )

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_operation(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        copied["operation_type"] = _coerce_enum(
            copied.get(
                "operation_type",
                "UNKNOWN",
            ),
            VALID_OPERATION_TYPES,
            "UNKNOWN",
        )

        if not copied.get("id"):
            name = copied.get(
                "name",
                "operation",
            )

            copied["id"] = (
                _normalize_identifier(
                    str(name)
                )
                or "operation"
            )

        if "step" in copied and "order" not in copied:
            copied["order"] = copied["step"]
        elif "order" in copied and "step" not in copied:
            copied["step"] = copied["order"]

        if "operation" in copied and "name" not in copied:
            copied["name"] = str(copied["operation"])
        elif "name" in copied and "operation" not in copied:
            copied["operation"] = str(copied["name"])

        if "input" in copied and "inputs" not in copied:
            inp_val = copied["input"]
            copied["inputs"] = inp_val if isinstance(inp_val, list) else [str(inp_val)]
        elif "inputs" in copied and "input" not in copied:
            inps = copied["inputs"]
            copied["input"] = ", ".join(str(i) for i in inps) if len(inps) > 1 else (str(inps[0]) if inps else "x")

        if "output" in copied and "outputs" not in copied:
            out_val = copied["output"]
            copied["outputs"] = out_val if isinstance(out_val, list) else [str(out_val)]
        elif "outputs" in copied and "output" not in copied:
            outs = copied["outputs"]
            copied["output"] = ", ".join(str(o) for o in outs) if len(outs) > 1 else (str(outs[0]) if outs else "out")

        if "inputs" in copied:
            copied["inputs"] = _coerce_str_list(copied["inputs"])
        if "outputs" in copied:
            copied["outputs"] = _coerce_str_list(copied["outputs"])

        return copied

    @field_validator("inputs", "outputs", mode="before")
    @classmethod
    def _sanitize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_str_list(value)


# ---------------------------------------------------------------------
# Architecture connections
# ---------------------------------------------------------------------

class ArchitectureConnection(SafeBaseModel):
    """
    Directed relationship between components or operations.
    """

    source: str

    target: str

    connection_type: ConnectionType = "SEQUENTIAL"

    tensor: str | None = None

    order: int | None = Field(
        default=None,
        ge=0,
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_connection(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        copied["connection_type"] = _coerce_enum(
            copied.get(
                "connection_type",
                "SEQUENTIAL",
            ),
            VALID_CONNECTION_TYPES,
            "UNKNOWN",
        )

        return copied


# ---------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------

class ModelComponent(SafeBaseModel):
    """
    A model component with optional graph-level semantics.

    Backward compatible with the old:
        name
        component_type
        facts
    """

    id: str | None = None

    name: str

    component_type: str | None = None

    role: ComponentRole = "UNKNOWN"

    parent_id: str | None = None

    variant: str | None = None

    inputs: list[str] = Field(
        default_factory=list
    )

    outputs: list[str] = Field(
        default_factory=list
    )

    operations: list[str] = Field(
        default_factory=list
    )

    repeat_count: int | str | None = None

    dependencies: list[str] = Field(
        default_factory=list
    )

    subcomponents: list["ModelComponent"] = Field(
        default_factory=list
    )

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_component(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        copied["role"] = _coerce_enum(
            copied.get(
                "role",
                "UNKNOWN",
            ),
            VALID_COMPONENT_ROLES,
            "UNKNOWN",
        )

        if isinstance(
            copied.get("name"),
            str,
        ):
            copied["name"] = (
                _normalize_space(
                    copied["name"]
                )
            )

        if not copied.get("id"):
            name = str(
                copied.get(
                    "name",
                    "component",
                )
            )

            variant = copied.get(
                "variant"
            )

            identifier = name

            if variant:
                identifier = (
                    f"{identifier}_{variant}"
                )

            copied["id"] = (
                _normalize_identifier(
                    identifier
                )
                or "component"
            )

        if "inputs" in copied:
            copied["inputs"] = _coerce_str_list(copied["inputs"])
        if "outputs" in copied:
            copied["outputs"] = _coerce_str_list(copied["outputs"])
        if "depends_on" in copied and "dependencies" not in copied:
            copied["dependencies"] = copied.pop("depends_on")
        if "dependencies" in copied:
            copied["dependencies"] = _coerce_str_list(copied["dependencies"])

        return copied

    @field_validator("inputs", "outputs", "operations", "dependencies", mode="before")
    @classmethod
    def _sanitize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_str_list(value)


# ---------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------

class ModelVariant(SafeBaseModel):
    """
    Explicit model-scale/configuration variant.

    Example:
        PointDiT-B
        PointDiT-L
        PointDiT-H
    """

    id: str

    name: str

    base_model: str | None = None

    component_ids: list[str] = Field(
        default_factory=list
    )

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_variant(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        if not copied.get("id"):
            copied["id"] = (
                _normalize_identifier(
                    str(
                        copied.get(
                            "name",
                            "variant",
                        )
                    )
                )
                or "variant"
            )

        return copied


# ---------------------------------------------------------------------
# Architecture graph
# ---------------------------------------------------------------------

class ArchitectureNode(SafeBaseModel):
    """
    Represent a node in the architecture graph (component / layer / module).
    """

    id: str
    type: str = "Unknown"
    name: str | None = None
    inputs: list[str] = Field(
        default_factory=list
    )
    outputs: list[str] = Field(
        default_factory=list
    )
    facts: list[ImplementationFact] = Field(
        default_factory=list
    )
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_node(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)
        if not copied.get("id"):
            name = copied.get("name") or copied.get("id") or "node"
            copied["id"] = _normalize_identifier(str(name)) or "node"
        if not copied.get("type"):
            copied["type"] = copied.get("component_type") or copied.get("role") or "Unknown"
        return copied


class ArchitectureEdge(SafeBaseModel):
    """
    Represent a directed edge in the architecture graph.
    """

    from_node: str = Field(
        default="",
        alias="from",
    )
    to_node: str = Field(
        default="",
        alias="to",
    )
    connection_type: ConnectionType = "SEQUENTIAL"
    tensor: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_edge(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)
        if "from" not in copied:
            if "from_node" in copied:
                copied["from"] = copied["from_node"]
            elif "source" in copied:
                copied["from"] = copied["source"]
        if "to" not in copied:
            if "to_node" in copied:
                copied["to"] = copied["to_node"]
            elif "target" in copied:
                copied["to"] = copied["target"]

        copied["connection_type"] = _coerce_enum(
            copied.get("connection_type", "SEQUENTIAL"),
            VALID_CONNECTION_TYPES,
            "SEQUENTIAL",
        )
        return copied

    @model_serializer(mode="wrap")
    def serialize_edge(self, handler, info):
        data = handler(self)
        if isinstance(data, dict):
            if "from_node" in data:
                data["from"] = data.pop("from_node")
            if "to_node" in data:
                data["to"] = data.pop("to_node")
        return data

    @property
    def source(self) -> str:
        return self.from_node

    @property
    def target(self) -> str:
        return self.to_node


class ArchitectureGraph(SafeBaseModel):
    """
    Explicit forward architecture representation.
    """

    nodes: list[ArchitectureNode] = Field(
        default_factory=list
    )

    edges: list[ArchitectureEdge] = Field(
        default_factory=list
    )

    inputs: list[str] = Field(
        default_factory=list
    )

    outputs: list[str] = Field(
        default_factory=list
    )

    tensors: list[TensorSpec] = Field(
        default_factory=list
    )

    operations: list[ArchitectureOperation] = Field(
        default_factory=list
    )

    connections: list[ArchitectureConnection] = Field(
        default_factory=list
    )

    forward_pass: list[ForwardPassStep] = Field(
        default_factory=list
    )

    tensor_flow: str | None = None

    primary_topology_source: str = "TEXT"

    branches: list[str] = Field(
        default_factory=list
    )

    skips: list[str] = Field(
        default_factory=list
    )

    residuals: list[str] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_architecture_graph(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        if "skip_connections" in copied and "skips" not in copied:
            copied["skips"] = copied.pop("skip_connections")
        if "residual_connections" in copied and "residuals" not in copied:
            copied["residuals"] = copied.pop("residual_connections")
        if "branching" in copied and "branches" not in copied:
            copied["branches"] = copied.pop("branching")

        for k in ("branches", "skips", "residuals"):
            if k in copied:
                copied[k] = _coerce_str_list(copied[k])
        return copied

    def get_tensor_flow_chain(self) -> str:
        """
        Derive directed tensor flow chain string.
        Example: 'Image -> PatchEmbedding -> Tokens -> Transformer -> Logits'
        """
        if self.tensor_flow:
            return self.tensor_flow

        chain_parts: list[str] = []
        for step in self.forward_pass:
            inp = str(step.input).strip()
            op = str(step.operation).strip()
            out = str(step.output).strip()
            if not chain_parts:
                chain_parts.extend([inp, op, out])
            else:
                if chain_parts[-1] == inp:
                    chain_parts.extend([op, out])
                else:
                    chain_parts.extend([f"[{inp}]", op, out])
        if not chain_parts and self.edges:
            edge_strs = [f"{e.from_node} -> {e.to_node}" for e in self.edges]
            return " -> ".join(edge_strs)
        return " -> ".join(chain_parts)

    @field_validator("inputs", "outputs", mode="before")
    @classmethod
    def _sanitize_string_lists(cls, value: Any) -> list[str]:
        return _coerce_str_list(value)

    @model_validator(mode="before")
    @classmethod
    def _sanitize_graph_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            copied = dict(data)
            if "inputs" in copied:
                copied["inputs"] = _coerce_str_list(copied["inputs"])
            if "outputs" in copied:
                copied["outputs"] = _coerce_str_list(copied["outputs"])
            return copied
        return data

    @model_validator(mode="after")
    def validate_graph_references(
        self,
    ) -> "ArchitectureGraph":
        """
        Keep graph validation conservative.

        We do not reject unresolved references because papers often omit
        implementation details. Instead, we preserve the graph and let
        the verification pipeline classify uncertainty.
        """
        return self


ModelComponent.model_rebuild()


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------

class TrainingStage(SafeBaseModel):
    """
    Stage-specific training configuration.

    Prevents:
        Stage 2 resolution = 512
    from becoming:
        universal input_shape = 512 x 512
    """

    id: str

    name: str

    order: int | None = Field(
        default=None,
        ge=1,
    )

    resolution: ShapeSpec | None = None

    epochs: int | str | None = None

    optimizer: str | None = None

    base_learning_rate: float | str | None = None

    effective_learning_rate: float | str | None = None

    learning_rate_scaling_rule: str | None = None

    batch_size: int | str | None = None

    datasets: list[str] = Field(
        default_factory=list
    )

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_stage(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        if not copied.get("id"):
            copied["id"] = (
                _normalize_identifier(
                    str(
                        copied.get(
                            "name",
                            "training_stage",
                        )
                    )
                )
                or "training_stage"
            )

        return copied


# ---------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------

class InferenceSpec(SafeBaseModel):
    sampler: str | None = None

    solver: str | None = None

    step_counts: list[int] = Field(
        default_factory=list
    )

    initialization: list[str] = Field(
        default_factory=list
    )

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None


# ---------------------------------------------------------------------
# Conflict tracking
# ---------------------------------------------------------------------

class ConflictRecord(SafeBaseModel):
    """
    Explicit unresolved disagreement between extracted claims.
    """

    subject: str

    values: list[Any] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    resolution: str | None = None

    status: EvidenceStatus = "AMBIGUOUS"

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_conflict(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        copied = dict(data)

        copied["status"] = _coerce_enum(
            copied.get(
                "status",
                "AMBIGUOUS",
            ),
            VALID_STATUSES,
            "AMBIGUOUS",
        )

        return copied


class ExecutableOperation(SafeBaseModel):
    """
    An equation extracted as a deterministic executable operation.
    Example:
        {"operation": "attention", "formula": "softmax(QK^T/sqrt(dk))V"}
    """

    operation: str
    formula: str
    code_expression: str | None = None
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_equation(cls, data: Any) -> Any:
        if isinstance(data, str):
            text = data.strip()
            if ":" in text:
                parts = text.split(":", 1)
                return {"operation": parts[0].strip(), "formula": parts[1].strip()}
            if "=" in text:
                parts = text.split("=", 1)
                return {"operation": parts[0].strip(), "formula": parts[1].strip()}
            return {"operation": "operation", "formula": text}
        return data


class ArchitectureSpec(SafeBaseModel):
    """
    Architectural specification scoped exclusively for models.py generation.
    Omits training, preprocessing, and evaluation details to avoid distraction.
    """

    paper_title: str | None = None
    task: str | None = None
    model_components: list[ModelComponent] = Field(default_factory=list)
    architecture: ArchitectureGraph = Field(default_factory=ArchitectureGraph)
    architecture_graph: ArchitectureGraph = Field(default_factory=ArchitectureGraph)
    forward_pass: list[ForwardPassStep] = Field(default_factory=list)
    tensor_flow: str | None = None
    equations: list[ExecutableOperation] = Field(default_factory=list)


class TrainingSpec(SafeBaseModel):
    """Training specification scoped exclusively for train.py generation."""

    paper_title: str | None = None
    training: list[ImplementationFact] = Field(default_factory=list)
    training_stages: list[TrainingStage] = Field(default_factory=list)


class DatasetSpec(SafeBaseModel):
    """Dataset and preprocessing specification scoped exclusively for dataset.py generation."""

    paper_title: str | None = None
    preprocessing: list[ImplementationFact] = Field(default_factory=list)


class EvaluationSpec(SafeBaseModel):
    """Evaluation specification scoped exclusively for inference and evaluation generation."""

    paper_title: str | None = None
    inference: InferenceSpec = Field(default_factory=InferenceSpec)
    facts: list[ImplementationFact] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Final implementation specification
# ---------------------------------------------------------------------

class ImplementationSpec(SafeBaseModel):
    """
    Complete paper implementation specification.

    Backward-compatible fields:
        paper_title
        task
        model_components
        preprocessing
        training
        unknowns

    New structured fields:
        architecture
        variants
        training_stages
        inference
        conflicts
    """

    schema_version: str = "2.0"

    paper_title: str | None = None

    task: str | None = None

    model_components: list[ModelComponent] = Field(
        default_factory=list
    )

    architecture: ArchitectureGraph = Field(
        default_factory=ArchitectureGraph
    )

    architecture_graph: ArchitectureGraph = Field(
        default_factory=ArchitectureGraph
    )

    forward_pass: list[ForwardPassStep] = Field(
        default_factory=list
    )

    tensor_flow: str | None = None

    equations: list[ExecutableOperation] = Field(
        default_factory=list
    )

    variants: list[ModelVariant] = Field(
        default_factory=list
    )

    preprocessing: list[ImplementationFact] = Field(
        default_factory=list
    )

    training: list[ImplementationFact] = Field(
        default_factory=list
    )

    training_stages: list[TrainingStage] = Field(
        default_factory=list
    )

    inference: InferenceSpec = Field(
        default_factory=InferenceSpec
    )

    unknowns: list[ImplementationFact] = Field(
        default_factory=list
    )

    conflicts: list[ConflictRecord] = Field(
        default_factory=list
    )

    notes: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_spec(
        cls,
        data: Any,
    ) -> Any:
        # Preserve compatibility with the old behavior where a bare
        # component list could be validated as an ImplementationSpec.
        if isinstance(data, list):
            return {
                "model_components": data
            }

        if not isinstance(data, dict):
            return data

        copied = dict(data)

        if not copied.get("schema_version"):
            copied["schema_version"] = "2.0"

        if "architecture_graph" in copied and "architecture" not in copied:
            copied["architecture"] = copied["architecture_graph"]
        elif "architecture" in copied and "architecture_graph" not in copied:
            copied["architecture_graph"] = copied["architecture"]

        return copied

    @model_validator(mode="after")
    def validate_spec_consistency(
        self,
    ) -> "ImplementationSpec":
        """
        Apply narrow consistency checks without inventing facts.
        """

        # ---------------------------------------------------------
        # 1. UNKNOWN facts should not also claim perfect certainty.
        # ---------------------------------------------------------
        for fact in self._all_facts():
            if fact.status == "UNKNOWN":
                fact.confidence = min(
                    fact.confidence,
                    0.5,
                )

        # ---------------------------------------------------------
        # 2. Deduplicate notes while preserving order.
        # ---------------------------------------------------------
        seen_notes: set[str] = set()
        unique_notes: list[str] = []

        for note in self.notes:
            normalized = note.strip()

            if (
                not normalized
                or normalized in seen_notes
            ):
                continue

            seen_notes.add(normalized)
            unique_notes.append(normalized)

        self.notes = unique_notes

        # ---------------------------------------------------------
        # 3. Synchronize and populate architecture graph topology.
        # ---------------------------------------------------------
        nodes = list(self.architecture_graph.nodes) or list(self.architecture.nodes)
        edges = list(self.architecture_graph.edges) or list(self.architecture.edges)
        tensors = list(self.architecture_graph.tensors) or list(self.architecture.tensors)

        existing_node_ids = {n.id for n in nodes}

        for comp in self.model_components:
            node_id = comp.id or _normalize_identifier(comp.name) or comp.name
            if node_id not in existing_node_ids:
                node_type = comp.component_type or str(comp.role) or "Unknown"
                nodes.append(
                    ArchitectureNode(
                        id=node_id,
                        type=node_type,
                        name=comp.name,
                        inputs=list(comp.inputs),
                        outputs=list(comp.outputs),
                        facts=list(comp.facts),
                    )
                )
                existing_node_ids.add(node_id)

            if comp.parent_id:
                edges.append(
                    ArchitectureEdge(from_node=comp.parent_id, to_node=node_id)
                )
            for inp in comp.inputs:
                edges.append(
                    ArchitectureEdge(from_node=inp, to_node=node_id)
                )

        for conn in self.architecture.connections:
            edges.append(
                ArchitectureEdge(from_node=conn.source, to_node=conn.target)
            )

        self.architecture_graph.nodes = nodes
        self.architecture_graph.edges = edges
        self.architecture_graph.tensors = tensors
        self.architecture.nodes = nodes
        self.architecture.edges = edges
        self.architecture.tensors = tensors

        # ---------------------------------------------------------
        # 4. Synchronize and populate explicit forward pass contract.
        # ---------------------------------------------------------
        fps = (
            list(self.forward_pass)
            or list(self.architecture_graph.forward_pass)
            or list(self.architecture.forward_pass)
        )

        if not fps:
            ops = self.architecture.operations or self.architecture_graph.operations
            if ops:
                for idx, op in enumerate(ops, start=1):
                    step_num = op.step if op.step is not None else (op.order if op.order is not None else idx)
                    op_name = op.operation or op.name or op.id or "layer"
                    inp = op.input if op.input is not None else (", ".join(op.inputs) if len(op.inputs) > 1 else (op.inputs[0] if op.inputs else "x"))
                    out = op.output if op.output is not None else (", ".join(op.outputs) if len(op.outputs) > 1 else (op.outputs[0] if op.outputs else "out"))
                    fps.append(
                        ForwardPassStep(
                            step=step_num,
                            operation=op_name,
                            input=inp,
                            output=out,
                        )
                    )
            elif self.model_components:
                for idx, comp in enumerate(self.model_components, start=1):
                    inps = list(comp.inputs)
                    outs = list(comp.outputs)
                    inp = ", ".join(inps) if len(inps) > 1 else (inps[0] if inps else (f"out_{idx-1}" if idx > 1 else "x"))
                    out = ", ".join(outs) if len(outs) > 1 else (outs[0] if outs else f"out_{idx}")
                    fps.append(
                        ForwardPassStep(
                            step=idx,
                            operation=comp.name or comp.id or "layer",
                            input=inp,
                            output=out,
                        )
                    )

        for idx, step in enumerate(fps):
            if not step.consumer_operation and idx + 1 < len(fps):
                step.consumer_operation = fps[idx + 1].operation

        self.forward_pass = fps
        self.architecture_graph.forward_pass = fps
        self.architecture.forward_pass = fps

        # ---------------------------------------------------------
        # 5. Synchronize and populate preserved tensor flow chain.
        # ---------------------------------------------------------
        tf = (
            self.tensor_flow
            or self.architecture_graph.tensor_flow
            or self.architecture.tensor_flow
            or self.architecture_graph.get_tensor_flow_chain()
        )

        self.tensor_flow = tf
        self.architecture_graph.tensor_flow = tf
        self.architecture.tensor_flow = tf

        return self

    def _all_facts(
        self,
    ) -> list[ImplementationFact]:
        facts: list[ImplementationFact] = []

        facts.extend(
            self.preprocessing
        )

        facts.extend(
            self.training
        )

        facts.extend(
            self.unknowns
        )

        facts.extend(
            self.inference.facts
        )

        for component in self.model_components:
            facts.extend(
                component.facts
            )

        for variant in self.variants:
            facts.extend(
                variant.facts
            )

        for stage in self.training_stages:
            facts.extend(
                stage.facts
            )

        for tensor in self.architecture.tensors:
            facts.extend(
                tensor.facts
            )

        for operation in self.architecture.operations:
            facts.extend(
                operation.facts
            )

        for node in self.architecture_graph.nodes:
            facts.extend(
                node.facts
            )

        return facts

    def get_unresolved_required_facts(self) -> list[ImplementationFact]:
        """Return all required facts whose status/value remains unresolved."""
        unresolved: list[ImplementationFact] = []
        for fact in self._all_facts():
            if fact.required:
                if (
                    fact.value is None
                    or fact.value == "UNKNOWN"
                    or fact.status == "UNKNOWN"
                ):
                    unresolved.append(fact)
        return unresolved

    def validate_ready_for_generation(self) -> bool:
        """
        Refuse to start generation if required facts are unresolved.
        Raises ValueError if any required fact is unresolved.
        """
        unresolved = self.get_unresolved_required_facts()
        if unresolved:
            names = sorted({f.name for f in unresolved})
            raise ValueError(
                f"Generation refused to start: unresolved required facts: {names}"
            )
        return True

    def get_topological_generation_order(self) -> list[ModelComponent]:
        """
        Return model components sorted in deterministic bottom-up topological
        dependency order (primitives and subcomponents before container models).
        """
        all_comps: list[ModelComponent] = []
        def _collect(comps: list[ModelComponent]):
            for c in comps:
                all_comps.append(c)
                if c.subcomponents:
                    _collect(c.subcomponents)
        _collect(self.model_components)

        name_to_comp = {c.name: c for c in all_comps}
        deps: dict[str, set[str]] = {c.name: set(c.dependencies) for c in all_comps}
        for c in all_comps:
            for sc in c.subcomponents:
                deps[c.name].add(sc.name)

        visited: set[str] = set()
        order: list[ModelComponent] = []
        def _visit(name: str, visiting: set[str]):
            if name in visited or name in visiting:
                return
            visiting.add(name)
            for d in sorted(deps.get(name, set())):
                if d in name_to_comp:
                    _visit(d, visiting)
            visiting.remove(name)
            visited.add(name)
            if name in name_to_comp:
                order.append(name_to_comp[name])

        for c in all_comps:
            _visit(c.name, set())
        return order

    def to_architecture_spec(self) -> ArchitectureSpec:
        return ArchitectureSpec(
            paper_title=self.paper_title,
            task=self.task,
            model_components=self.model_components,
            architecture=self.architecture,
            architecture_graph=self.architecture_graph,
            forward_pass=self.forward_pass,
            tensor_flow=self.tensor_flow,
            equations=self.equations,
        )

    def to_training_spec(self) -> TrainingSpec:
        return TrainingSpec(
            paper_title=self.paper_title,
            training=self.training,
            training_stages=self.training_stages,
        )

    def to_dataset_spec(self) -> DatasetSpec:
        return DatasetSpec(
            paper_title=self.paper_title,
            preprocessing=self.preprocessing,
        )

    def to_evaluation_spec(self) -> EvaluationSpec:
        return EvaluationSpec(
            paper_title=self.paper_title,
            inference=self.inference,
        )