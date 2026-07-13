from adapters.understanding_adapter import spec_from_understanding


def test_spec_from_understanding_includes_architecture_graph():
    data = {
        "paper_title": "Graph Paper",
        "task": "Classification",
        "model_components": [
            {
                "id": "encoder",
                "name": "Encoder",
                "component_type": "Encoder"
            }
        ],
        "architecture_graph": {
            "nodes": [
                {"id": "encoder", "type": "Encoder"}
            ],
            "edges": [
                {"from": "embedding", "to": "encoder"}
            ],
            "tensors": []
        }
    }
    result = spec_from_understanding(data)
    assert "## Architecture Graph Topology" in result
    assert "### Nodes" in result
    assert "- encoder: Encoder" in result
    assert "### Edges" in result
    assert "- embedding -> encoder" in result


def test_spec_from_understanding_includes_forward_pass_contract():
    data = {
        "paper_title": "Forward Pass Paper",
        "forward_pass": [
            {
                "step": 3,
                "operation": "encoder",
                "input": "embedded_tokens",
                "output": "memory"
            }
        ]
    }
    result = spec_from_understanding(data)
    assert "## Forward Pass Contract" in result
    assert "memory = self.encoder(embedded_tokens)" in result
    assert '{"step": 3, "operation": "encoder", "input": "embedded_tokens", "output": "memory"}' in result


def test_spec_from_understanding_includes_preserved_tensor_flow():
    data = {
        "paper_title": "Tensor Flow Architecture",
        "tensor_flow": "Image -> PatchEmbedding -> Tokens -> Transformer -> Logits"
    }
    result = spec_from_understanding(data)
    assert "## Preserved Tensor Flow (Architecture)" in result
    assert "Image -> PatchEmbedding -> Tokens -> Transformer -> Logits" in result


def test_executable_equation_operations():
    data = {
        "paper_title": "Attention Paper",
        "equations": [
            {
                "operation": "attention",
                "formula": "softmax(QK^T/sqrt(dk))V"
            }
        ]
    }
    result = spec_from_understanding(data)
    assert "## Executable Equation Operations" in result
    assert "operation: attention" in result
    assert "formula: softmax(QK^T/sqrt(dk))V" in result
    assert '{"operation": "attention", "formula": "softmax(QK^T/sqrt(dk))V"}' in result


def test_architecture_spec_separation_for_models_py():
    data = {
        "paper_title": "Split Specs Paper",
        "tensor_flow": "Input -> Conv -> Output",
        "preprocessing": [{"name": "image_size", "value": 224}],
        "training": [{"name": "optimizer", "value": "AdamW"}],
    }
    result_models = spec_from_understanding(data, target_file="models.py")
    assert "## Preserved Tensor Flow (Architecture)" in result_models
    assert "## Preprocessing Steps" not in result_models
    assert "## Training Configuration" not in result_models


def test_component_hierarchy():
    data = {
        "paper_title": "Transformer Hierarchy",
        "model_components": [
            {
                "name": "Transformer",
                "subcomponents": [
                    {
                        "name": "Encoder",
                        "subcomponents": [
                            {
                                "name": "EncoderLayer",
                                "repeat_count": 6,
                                "subcomponents": [
                                    {"name": "Attention"},
                                    {"name": "LayerNorm"},
                                    {"name": "FeedForward"},
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    result = spec_from_understanding(data)
    assert "## Component Hierarchy" in result
    assert "EncoderLayer × 6" in result
    assert "Attention" in result


def test_execution_graph_consumer_operation():
    data = {
        "paper_title": "Execution Graph",
        "forward_pass": [
            {
                "step": 1,
                "operation": "encoder",
                "input": "embedded_tokens",
                "output": "memory",
                "consumer_operation": "decoder"
            }
        ]
    }
    result = spec_from_understanding(data)
    assert "## Forward Pass Contract (Execution Graph)" in result
    assert "Input: embedded_tokens -> Operation: encoder -> Output: memory -> Consumer: decoder" in result


def test_figure_primary_topology_source():
    data = {
        "paper_title": "Figure Topology Paper",
        "architecture_graph": {
            "primary_topology_source": "FIGURE",
            "branches": ["class_token -> head"],
            "skips": ["x -> add_1"],
            "residuals": ["attn_out + x"],
        }
    }
    result = spec_from_understanding(data)
    assert "Primary Topology Source: FIGURE" in result
    assert "### Branches" in result
    assert "class_token -> head" in result
    assert "### Skip Connections" in result
    assert "x -> add_1" in result
    assert "### Residual Connections" in result
    assert "attn_out + x" in result





