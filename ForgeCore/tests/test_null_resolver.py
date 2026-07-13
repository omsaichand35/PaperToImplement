from null_resolver import resolve_null_entities
from models import ImplementationSpec, ModelComponent, ImplementationFact


def test_null_resolver_resolves_required_component_facts():
    spec = ImplementationSpec(
        paper_title="Component Gaps Paper",
        task="Image Classification",
        model_components=[
            ModelComponent(
                name="Backbone",
                component_type="CNN",
                facts=[
                    ImplementationFact(
                        name="num_layers",
                        value=None,
                        required=True,
                        status="UNKNOWN",
                    )
                ],
            )
        ],
    )

    report = resolve_null_entities(spec, dataset_name="CIFAR10")

    assert report["resolved_count"] > 0
    assert spec.model_components[0].facts[0].value == 6
    assert spec.model_components[0].facts[0].status == "DOMAIN_HEURISTIC"
