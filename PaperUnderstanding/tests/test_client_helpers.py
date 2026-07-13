import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from client import (  # noqa: E402
    extract_json_payload,
    merge_specs,
    split_pages_into_chunks,
    _sanitize_spec_payload_dict,
    merge_architecture_graphs,
)
from models import (  # noqa: E402
    Evidence,
    ImplementationFact,
    ImplementationSpec,
    ModelComponent,
    ArchitectureGraph,
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


import pytest
import asyncio
from client import analyze_visual_assets, validate_spec_payload


def test_analyze_visual_assets_empty_when_disabled():
    result = asyncio.run(
        analyze_visual_assets(
            pages=[{"page": 1, "images": [{"image_id": "p1_img1"}]}],
            schema="{}",
            enable_vision=False,
        )
    )
    assert result == []


def test_analyze_visual_assets_empty_when_no_images():
    result = asyncio.run(
        analyze_visual_assets(
            pages=[{"page": 1, "images": []}],
            schema="{}",
            enable_vision=True,
        )
    )
    assert result == []


def test_validate_spec_payload_coerces_dict_inputs_outputs():
    raw_data = {
        "paper_title": "AlexNet",
        "task": "Image Classification",
        "architecture": {
            "inputs": [
                {"name": "input_image", "dimensions": ["H", "W", 3]}
            ],
            "outputs": [
                {"name": "output_prob", "dimensions": ["N", 1000]}
            ],
        },
    }
    spec = validate_spec_payload(raw_data, context="Test payload")
    assert spec.architecture.inputs[0] == "input_image ['H', 'W', 3]"
    assert spec.architecture.outputs[0] == "output_prob ['N', 1000]"


def test_validate_spec_payload_coerces_str_operations():
    raw_data = {
        "paper_title": "AlexNet",
        "task": "Image Classification",
        "architecture": {
            "operations": [
                "Convolutional Layer 1",
                "ReLU Activation 1",
                "Max Pooling 1",
            ],
        },
    }
    spec = validate_spec_payload(raw_data, context="Test op payload")
    assert len(spec.architecture.operations) == 3
    assert spec.architecture.operations[0].name == "Convolutional Layer 1"
    assert spec.architecture.operations[1].name == "ReLU Activation 1"
    assert spec.architecture.operations[0].operation_type == "UNKNOWN"


def test_validate_spec_payload_coerces_list_inference():
    raw_data = {
        "paper_title": "AlexNet",
        "task": "Image Classification",
        "inference": [
            {"name": "inference_time", "value": "1.2ms", "status": "PAPER_REPORTED", "evidence": []},
            {"name": "throughput", "value": "1000 img/s", "status": "UNKNOWN", "evidence": []},
        ],
    }
    spec = validate_spec_payload(raw_data, context="Test inference list payload")
    # Should not raise; inference.facts contains the folded items
    assert len(spec.inference.facts) == 2
    assert spec.inference.facts[0].name == "inference_time"


def test_validate_spec_payload_coerces_dict_training():
    raw_data = {
        "paper_title": "AlexNet",
        "task": "Image Classification",
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
        },
    }
    spec = validate_spec_payload(raw_data, context="Test training dict payload")
    assert len(spec.training) == 2
    assert any(f.name == "optimizer" and f.value == "AdamW" for f in spec.training)
    assert any(f.name == "learning_rate" and f.value == 0.001 for f in spec.training)


def test_validate_spec_payload_coerces_dict_components():
    raw_data = {
        "paper_title": "AlexNet",
        "task": "Image Classification",
        "model_components": {
            "backbone": {
                "component_type": "BACKBONE",
                "facts": [
                    {"name": "layers", "value": 50, "status": "PAPER_REPORTED"}
                ]
            }
        },
    }
    spec = validate_spec_payload(raw_data, context="Test components dict payload")
    assert len(spec.model_components) == 1
    assert spec.model_components[0].name == "backbone"
    assert spec.model_components[0].component_type == "BACKBONE"
    assert spec.model_components[0].facts[0].name == "layers"
    assert spec.model_components[0].facts[0].value == 50


def test_architecture_graph_topology():
    raw_data = {
        "paper_title": "Topology Test",
        "model_components": [
            {
                "id": "embedding",
                "name": "Patch Embedding",
                "component_type": "Embedding"
            },
            {
                "id": "encoder",
                "name": "Transformer Encoder",
                "component_type": "Encoder",
                "inputs": ["embedding"]
            }
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    assert len(spec.architecture_graph.nodes) == 2
    assert len(spec.architecture_graph.edges) == 1

    dumped = spec.model_dump()
    assert "architecture_graph" in dumped
    arch_graph = dumped["architecture_graph"]
    assert "nodes" in arch_graph
    assert "edges" in arch_graph
    assert "tensors" in arch_graph

    node_ids = {n["id"] for n in arch_graph["nodes"]}
    assert "encoder" in node_ids
    assert "embedding" in node_ids

    assert arch_graph["edges"][0]["from"] == "embedding"
    assert arch_graph["edges"][0]["to"] == "encoder"


def test_forward_pass_contract():
    raw_data = {
        "paper_title": "Forward Pass Test",
        "forward_pass": [
            {
                "step": 3,
                "operation": "Encoder",
                "input": "embedded_tokens",
                "output": "memory"
            }
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    assert len(spec.forward_pass) == 1
    step = spec.forward_pass[0]
    assert step.step == 3
    assert step.operation == "Encoder"
    assert step.input == "embedded_tokens"
    assert step.output == "memory"
    assert step.to_code_line() == "memory = self.encoder(embedded_tokens)"

    dumped = spec.model_dump()
    assert "forward_pass" in dumped
    assert dumped["forward_pass"][0] == {
        "step": 3,
        "operation": "Encoder",
        "input": "embedded_tokens",
        "output": "memory",
        "consumer_operation": None,
        "notes": None,
    }


def test_forward_pass_string_coercion():
    from client import _sanitize_spec_payload_dict
    raw = {
        "forward_pass": [
            "memory = self.encoder(embedded_tokens)"
        ]
    }
    sanitized = _sanitize_spec_payload_dict(raw)
    assert sanitized["forward_pass"][0] == {
        "step": 1,
        "operation": "encoder",
        "input": "embedded_tokens",
        "output": "memory",
    }


def test_tensor_flow_preservation():
    raw_data = {
        "paper_title": "Tensor Flow Paper",
        "forward_pass": [
            {"step": 1, "operation": "PatchEmbedding", "input": "Image", "output": "Tokens"},
            {"step": 2, "operation": "Transformer", "input": "Tokens", "output": "Logits"}
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    assert spec.tensor_flow == "Image -> PatchEmbedding -> Tokens -> Transformer -> Logits"
    assert spec.architecture_graph.get_tensor_flow_chain() == "Image -> PatchEmbedding -> Tokens -> Transformer -> Logits"


def test_required_vs_optional_facts():
    raw_data = {
        "paper_title": "Required Facts Paper",
        "preprocessing": [
            {"name": "layer count", "value": 12},
            {"name": "tensor shapes", "value": "[N, D]"},
            {"name": "optimizer", "value": "AdamW"},
            {"name": "epochs", "value": 100},
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    facts = {f.name: f.required for f in spec.preprocessing}
    assert facts["layer count"] is True
    assert facts["tensor shapes"] is True
    assert facts["optimizer"] is False
    assert facts["epochs"] is False


def test_generation_refusal_if_required_unresolved():
    import pytest
    raw_data = {
        "paper_title": "Unresolved Paper",
        "preprocessing": [
            {"name": "layer count", "value": None, "required": True},
            {"name": "epochs", "value": None, "required": False},
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    with pytest.raises(ValueError, match="Generation refused to start: unresolved required facts"):
        spec.validate_ready_for_generation()


def test_equation_string_coercion():
    raw_data = {
        "paper_title": "Equation Paper",
        "equations": [
            "attention: softmax(QK^T/sqrt(dk))V"
        ]
    }
    sanitized = _sanitize_spec_payload_dict(raw_data)
    assert sanitized["equations"][0] == {
        "operation": "attention",
        "formula": "softmax(QK^T/sqrt(dk))V",
    }
    spec = ImplementationSpec.model_validate(raw_data)
    assert spec.equations[0].operation == "attention"
    assert spec.equations[0].formula == "softmax(QK^T/sqrt(dk))V"


def test_spec_splitting():
    raw_data = {
        "paper_title": "Multi-Spec Paper",
        "tensor_flow": "X -> Y",
        "preprocessing": [{"name": "crop", "value": "224x224"}],
        "training": [{"name": "lr", "value": 0.001}],
    }
    spec = ImplementationSpec.model_validate(raw_data)
    arch_spec = spec.to_architecture_spec()
    assert arch_spec.tensor_flow == "X -> Y"
    assert not hasattr(arch_spec, "training")
    train_spec = spec.to_training_spec()
    assert len(train_spec.training) == 1
    dataset_spec = spec.to_dataset_spec()
    assert len(dataset_spec.preprocessing) == 1


def test_topological_generation_order():
    raw_data = {
        "paper_title": "Transformer Order",
        "model_components": [
            {"name": "Transformer", "dependencies": ["EncoderStack"]},
            {"name": "EncoderStack", "dependencies": ["EncoderLayer"]},
            {"name": "EncoderLayer", "dependencies": ["MultiHeadAttention"]},
            {"name": "MultiHeadAttention", "dependencies": ["ScaledDotProductAttention"]},
            {"name": "ScaledDotProductAttention"},
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    order = [comp.name for comp in spec.get_topological_generation_order()]
    assert order.index("ScaledDotProductAttention") < order.index("MultiHeadAttention")
    assert order.index("MultiHeadAttention") < order.index("EncoderLayer")
    assert order.index("EncoderLayer") < order.index("EncoderStack")
    assert order.index("EncoderStack") < order.index("Transformer")


def test_auto_infer_consumer_operation():
    raw_data = {
        "paper_title": "Execution Graph Consumer",
        "forward_pass": [
            {"step": 1, "operation": "patch_embed", "input": "x", "output": "tokens"},
            {"step": 2, "operation": "transformer_block", "input": "tokens", "output": "features"},
        ]
    }
    spec = ImplementationSpec.model_validate(raw_data)
    spec.validate_spec_consistency()
    assert spec.forward_pass[0].consumer_operation == "transformer_block"


def test_merge_architecture_graphs_prioritizes_figure():
    text_graph = ArchitectureGraph(
        primary_topology_source="TEXT",
        tensor_flow="Text -> Only -> Flow",
    )
    fig_graph = ArchitectureGraph(
        primary_topology_source="FIGURE",
        tensor_flow="Image -> PatchEmbed -> Encoder -> Logits",
        branches=["class_token -> head"],
        skips=["x -> residual_add"],
    )
    merged = merge_architecture_graphs([text_graph, fig_graph])
    assert merged.primary_topology_source == "FIGURE"
    assert merged.tensor_flow == "Image -> PatchEmbed -> Encoder -> Logits"
    assert "class_token -> head" in merged.branches
    assert "x -> residual_add" in merged.skips








