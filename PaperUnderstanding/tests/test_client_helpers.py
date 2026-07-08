import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from client import (  # noqa: E402
    extract_json_payload,
    merge_specs,
    split_pages_into_chunks
)
from models import (  # noqa: E402
    Evidence,
    ImplementationFact,
    ImplementationSpec,
    ModelComponent
)


def test_split_pages_into_chunks():

    pages = [
        {"page": index, "text": f"text {index}"}
        for index in range(1, 10)
    ]

    chunks = split_pages_into_chunks(
        pages,
        pages_per_chunk=4
    )

    assert len(chunks) == 3
    assert [item["page"] for item in chunks[0]] == [1, 2, 3, 4]
    assert [item["page"] for item in chunks[1]] == [5, 6, 7, 8]
    assert [item["page"] for item in chunks[2]] == [9]


def test_extract_json_payload_from_code_fence():

    payload = extract_json_payload(
        """```json
{"paper_title": "Example", "task": null, "model_components": [], "preprocessing": [], "training": [], "unknowns": []}
```"""
    )

    assert payload["paper_title"] == "Example"


def test_merge_specs_combines_components_and_facts():

    first = ImplementationSpec(
        paper_title="Example",
        task="Task",
        model_components=[
            ModelComponent(
                name="Encoder",
                component_type="Transformer",
                facts=[
                    ImplementationFact(
                        name="layers",
                        value=12,
                        status="PAPER_REPORTED",
                        confidence=0.9,
                        evidence=[
                            Evidence(
                                page=2,
                                quote="12 layers"
                            )
                        ]
                    )
                ]
            )
        ]
    )

    second = ImplementationSpec(
        model_components=[
            ModelComponent(
                name="Encoder",
                component_type="Transformer",
                facts=[
                    ImplementationFact(
                        name="hidden_size",
                        value=768,
                        status="PAPER_REPORTED",
                        confidence=0.8,
                        evidence=[
                            Evidence(
                                page=3,
                                quote="hidden size 768"
                            )
                        ]
                    )
                ]
            )
        ]
    )

    merged = merge_specs([
        first,
        second
    ])

    assert merged.paper_title == "Example"
    assert merged.task == "Task"
    assert len(merged.model_components) == 1
    assert len(merged.model_components[0].facts) == 2


def test_implementation_spec_coerces_list():
    raw_data = [
        {"name": "PointDiT", "component_type": "Model"}
    ]
    spec = ImplementationSpec.model_validate(raw_data)
    assert len(spec.model_components) == 1
    assert spec.model_components[0].name == "PointDiT"
    assert spec.model_components[0].component_type == "Model"


def test_extract_json_payload_ignores_primitive_lists():
    payload = extract_json_payload(
        """Here are layers [2, 4, 6, 8] checked.
```json
{"paper_title": "Reviewed", "task": null, "model_components": [], "preprocessing": [], "training": [], "unknowns": []}
```"""
    )
    assert payload["paper_title"] == "Reviewed"
