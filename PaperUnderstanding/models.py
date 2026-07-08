from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


EvidenceStatus = Literal[
    "PAPER_REPORTED",
    "INFERRED",
    "ASSUMED",
    "UNKNOWN",
    "AMBIGUOUS",
    "REFERENCED_ELSEWHERE",
    "REGISTRY_CANONICAL",
    "LITERATURE_GROUNDED",
    "DOMAIN_HEURISTIC"
]


class SafeBaseModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def coerce_none_collections(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in list(data.items()):
                if v is None and k in cls.model_fields:
                    f_info = cls.model_fields[k]
                    annot = str(f_info.annotation) if f_info.annotation else ""
                    if "list" in annot or "List" in annot:
                        data[k] = []
                    elif "dict" in annot or "Dict" in annot:
                        data[k] = {}
        return data


class Evidence(SafeBaseModel):
    page: int | None = None
    section: str | None = None
    quote: str | None = None


class ImplementationFact(SafeBaseModel):
    name: str
    value: Any | None = None

    status: EvidenceStatus = "UNKNOWN"

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def sanitize_fact(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("status") not in {
                "PAPER_REPORTED", "INFERRED", "ASSUMED", "UNKNOWN", "AMBIGUOUS",
                "REFERENCED_ELSEWHERE", "REGISTRY_CANONICAL", "LITERATURE_GROUNDED",
                "DOMAIN_HEURISTIC"
            }:
                data["status"] = "UNKNOWN"
            if data.get("confidence") is None:
                data["confidence"] = 0.5
        return data


class ModelComponent(SafeBaseModel):
    name: str
    component_type: str | None = None

    facts: list[ImplementationFact] = Field(
        default_factory=list
    )


class ImplementationSpec(SafeBaseModel):
    paper_title: str | None = None

    task: str | None = None

    model_components: list[ModelComponent] = Field(
        default_factory=list
    )

    preprocessing: list[ImplementationFact] = Field(
        default_factory=list
    )

    training: list[ImplementationFact] = Field(
        default_factory=list
    )

    unknowns: list[ImplementationFact] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_spec(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"model_components": data}
        return data