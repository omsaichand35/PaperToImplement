from typing import Any, Callable


class UnderstandingAdapter:
    def __init__(
        self,
        analyze_paper_fn: Callable[..., Any],
    ):
        self.analyze_paper_fn = analyze_paper_fn

    async def analyze(
        self,
        pdf_path: str,
    ) -> Any:
        return await self.analyze_paper_fn(
            pdf_path=pdf_path
        )


def spec_from_understanding(spec_obj: Any) -> str:
    """
    Convert a PaperUnderstanding ImplementationSpec into a clean, structured
    specification string suitable for ForgeCore / ForgeCode.
    """
    if hasattr(spec_obj, "model_dump"):
        data = spec_obj.model_dump()
    elif isinstance(spec_obj, dict):
        data = spec_obj
    else:
        data = {}

    # Provenance status to human-readable source annotation
    _PROVENANCE_LABELS = {
        "REGISTRY_CANONICAL": "Source: torchvision canonical",
        "LITERATURE_GROUNDED": "Source: Literature consensus",
        "DOMAIN_HEURISTIC": "Source: Domain default",
    }

    title = data.get("paper_title") or "Research Paper Implementation"
    task = data.get("task") or "Deep learning model implementation"
    contract_lines = build_architecture_contract(data)

    lines = [
        f"# Project: {title}",
        f"Task: {task}\n",
        "## Architecture and Model Components",
    ]

    components = data.get("model_components", [])
    if not components:
        lines.append("- Implement model architecture as described in requirements.")
    for comp in components:
        c_name = comp.get("name") or "Component"
        c_type = comp.get("component_type") or "Module"
        lines.append(f"### {c_name} ({c_type})")
        for fact in comp.get("facts", []):
            f_name = fact.get("name") or ""
            f_val = fact.get("value")
            f_notes = fact.get("notes")
            f_status = fact.get("status") or ""
            f_evidence = fact.get("evidence", [])
            val_str = f": {f_val}" if f_val is not None else ""
            note_str = f" ({f_notes})" if f_notes else ""
            status_str = f" [{f_status}]" if f_status else ""
            lines.append(f"- {f_name}{val_str}{note_str}{status_str}")
            # Include direct quotes as implementation evidence
            for ev in f_evidence:
                quote = ev.get("quote")
                page = ev.get("page")
                if quote:
                    lines.append(f"  > Paper p.{page}: \"{quote}\"")
        lines.append("")

    prep = data.get("preprocessing", [])
    if prep:
        lines.append("## Preprocessing Steps")
        lines.append("Implement the following preprocessing pipeline:")
        for fact in prep:
            f_name = fact.get("name") or ""
            f_val = fact.get("value")
            f_status = fact.get("status") or ""
            val_str = f": {f_val}" if f_val is not None else ""
            provenance = _PROVENANCE_LABELS.get(f_status, "")
            prov_str = f" *({provenance})*" if provenance else ""
            lines.append(f"- {f_name}{val_str}{prov_str}")
            for ev in fact.get("evidence", []):
                quote = ev.get("quote")
                page = ev.get("page")
                if quote:
                    if page is not None:
                        lines.append(f"  > Paper p.{page}: \"{quote}\"")
                    else:
                        lines.append(f"  > {quote}")
        lines.append("")

    training = data.get("training", [])
    if training:
        lines.append("## Training Configuration")
        lines.append("Implement the following training setup:")
        for fact in training:
            f_name = fact.get("name") or ""
            f_val = fact.get("value")
            f_status = fact.get("status") or ""
            val_str = f": {f_val}" if f_val is not None else ""
            provenance = _PROVENANCE_LABELS.get(f_status, "")
            prov_str = f" *({provenance})*" if provenance else ""
            lines.append(f"- {f_name}{val_str}{prov_str}")
            for ev in fact.get("evidence", []):
                quote = ev.get("quote")
                page = ev.get("page")
                if quote:
                    if page is not None:
                        lines.append(f"  > Paper p.{page}: \"{quote}\"")
                    else:
                        lines.append(f"  > {quote}")
        lines.append("")

    # Collect names of facts already resolved into training/preprocessing
    resolved_fact_names: set[str] = set()
    for section_key in ("training", "preprocessing"):
        for fact in data.get(section_key, []):
            if fact.get("value") is not None:
                resolved_fact_names.add((fact.get("name") or "").lower().strip())

    unknowns = data.get("unknowns", [])
    # Filter out unknowns that have been resolved
    unresolved = [
        fact for fact in unknowns
        if (fact.get("name") or "").lower().strip() not in resolved_fact_names
    ]
    if unresolved:
        lines.append("## Unresolved Details")
        lines.append("The following details were not found in the paper:")
        for fact in unresolved:
            f_name = fact.get("name") or ""
            f_val = fact.get("value")
            f_notes = fact.get("notes")
            val_str = f": {f_val}" if f_val is not None else ""
            note_str = f" ({f_notes})" if f_notes else ""
            lines.append(f"- {f_name}{val_str}{note_str}")
        lines.append("")

    if contract_lines:
        lines.append("## Architecture Contract")
        lines.append(
            "The implementation must preserve these architecture-level invariants:"
        )
        lines.extend(contract_lines)
        lines.append("")

    lines.extend([
        "## Project Structure Requirements",
        "Generate the following files for a complete deep learning codebase:",
        "- models.py: Complete model architecture class with all layers, "
        "activations, normalization, pooling, dropout, and weight initialization.",
        "- dataset.py: Data loading, preprocessing pipeline, and data augmentation "
        "transforms as described above.",
        "- train.py: Full training loop with optimizer, learning rate scheduler, "
        "loss function, epoch iteration, checkpoint saving, and logging.",
        "- evaluate.py: Evaluation metrics (accuracy, top-k accuracy if applicable), "
        "model evaluation on validation/test set.",
        "- utils.py: Helper utilities (AverageMeter, checkpoint save/load, "
        "training logger).",
        "- tests/test_model.py: pytest test verifying forward pass produces "
        "expected output shape for dummy input.",
        "",
        "## Implementation Rules",
        "- Use PyTorch as the framework.",
        "- Store all hyperparameters as Python constants directly in the source files.",
        "- Do NOT create separate JSON or YAML config files.",
        "- Include all architecture details exactly as extracted from the paper.",
        "- Do not invent details not present in the specification above.",
    ])

    return "\n".join(lines)


def build_architecture_contract(data: dict) -> list[str]:
    """
    Convert extracted paper evidence into implementation
    invariants that survive planning and code generation.
    """

    facts = []

    for comp in data.get("model_components", []):
        facts.append(str(comp.get("name") or ""))
        facts.append(str(comp.get("component_type") or ""))
        for fact in comp.get("facts", []):
            facts.append(str(fact.get("name") or ""))
            facts.append(str(fact.get("value") or ""))
            facts.append(str(fact.get("notes") or ""))
            for ev in fact.get("evidence", []):
                facts.append(str(ev.get("quote") or ""))

    for section_key in ("preprocessing", "training", "unknowns"):
        for fact in data.get(section_key, []):
            facts.append(str(fact.get("name") or ""))
            facts.append(str(fact.get("value") or ""))
            facts.append(str(fact.get("notes") or ""))
            for ev in fact.get("evidence", []):
                facts.append(str(ev.get("quote") or ""))

    text = " ".join(facts).lower()
    contract: list[str] = []

    def add(line: str) -> None:
        if line not in contract:
            contract.append(line)

    if "dino" in text:
        add(
            "- If DINO/DINOv3 is specified, models.py must include a DINO feature-extractor branch and must not substitute it with a generic torchvision classifier."
        )

    if "diffusion" in text or "flow matching" in text:
        add(
            "- If diffusion or flow matching is specified, the model interface must include noisy sample and timestep inputs, and train.py must implement the paper's timestep/noise training objective."
        )

    if "patchify" in text or "patchification" in text or "unpatchify" in text:
        add(
            "- If patchify/unpatchify is specified, models.py must implement explicit patch/token embedding and reconstruction instead of a generic CNN head."
        )

    if "condition" in text and ("token" in text or "image" in text):
        add(
            "- If image conditioning is specified, models.py must show where conditioning features enter the model and how they are fused with the primary input."
        )

    if "point map normalization" in text or (
        "centroid" in text and "scale" in text and "point map" in text
    ):
        add(
            "- If point-map normalization is specified, dataset.py or utils.py must compute the paper's centroid/scale normalization before training."
        )

    if any(
        term in text
        for term in ("relp", "reld", "bf1", "affine-invariant", "delta")
    ):
        add(
            "- If dense geometry metrics are specified, evaluate.py must implement those metrics and must not use classification accuracy unless the task is classification."
        )

    if any(
        term in text
        for term in ("param", "parameters", "model configurations", "variant")
    ):
        add(
            "- If model variants or parameter counts are reported, keep the generated architecture in the same order of magnitude or record the mismatch as unresolved."
        )

    return contract
