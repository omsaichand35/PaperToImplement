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


def spec_dict_from_understanding(spec_obj: Any) -> dict:
    """
    Return the raw ImplementationSpec as a plain dict (lossless).
    This is what ForgeCode receives instead of the Markdown rendering.
    """
    if hasattr(spec_obj, "model_dump"):
        return spec_obj.model_dump()
    if isinstance(spec_obj, dict):
        return spec_obj
    return {}


def _fact_req_prefix(fact: dict[str, Any]) -> str:
    req = fact.get("required")
    if req is True or (isinstance(req, str) and req.lower() == "true"):
        return "[REQUIRED] "
    return "[OPTIONAL] "


def _format_component_hierarchy_lines(components: list[Any], indent: str = "") -> list[str]:
    lines = []
    for idx, comp in enumerate(components):
        if isinstance(comp, dict):
            name = comp.get("name") or comp.get("id") or "Component"
            repeat = comp.get("repeat_count")
            subs = comp.get("subcomponents", [])
        else:
            name = getattr(comp, "name", "") or getattr(comp, "id", "Component")
            repeat = getattr(comp, "repeat_count", None)
            subs = getattr(comp, "subcomponents", [])
        suffix = f" × {repeat}" if repeat and str(repeat) not in ("1", "None") else ""
        prefix = "└── " if idx == len(components) - 1 else "├── "
        lines.append(f"{indent}{prefix}{name}{suffix}")
        if subs:
            next_indent = indent + ("    " if idx == len(components) - 1 else "│   ")
            lines.extend(_format_component_hierarchy_lines(subs, next_indent))
    return lines


def spec_from_understanding(spec_obj: Any, target_file: str | None = None) -> str:
    """
    Convert a PaperUnderstanding ImplementationSpec into a clean, structured
    specification string suitable for ForgeCore / ForgeCode.
    If target_file is specified (e.g. 'models.py'), outputs only the relevant
    sub-spec (ArchitectureSpec) so the generator is not distracted.
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
            lines.append(f"- {_fact_req_prefix(fact)}{f_name}{val_str}{note_str}{status_str}")
            # Include direct quotes as implementation evidence
            for ev in f_evidence:
                quote = ev.get("quote")
                page = ev.get("page")
                if quote:
                    lines.append(f"  > Paper p.{page}: \"{quote}\"")
        lines.append("")

    has_hierarchy = any(
        (comp.get("subcomponents") if isinstance(comp, dict) else getattr(comp, "subcomponents", None))
        for comp in components
    )
    if has_hierarchy:
        lines.append("## Component Hierarchy")
        lines.extend(_format_component_hierarchy_lines(components))
        lines.append("")

    topological_order = []
    if hasattr(spec_obj, "get_topological_generation_order"):
        topological_order = spec_obj.get_topological_generation_order()
    elif any((comp.get("dependencies") if isinstance(comp, dict) else getattr(comp, "dependencies", None)) for comp in components):
        topological_order = components
    if topological_order:
        lines.append("## Topological Generation Order (Dependency Graph)")
        lines.append("Generate model components in this deterministic bottom-up dependency order:")
        for idx, comp in enumerate(topological_order, 1):
            cname = comp.get("name") if isinstance(comp, dict) else getattr(comp, "name", "Component")
            lines.append(f"{idx}. {cname}")
        lines.append("")

    arch_graph = data.get("architecture_graph") or data.get("architecture") or {}
    nodes = arch_graph.get("nodes", [])
    edges = arch_graph.get("edges", [])
    tensors = arch_graph.get("tensors", [])
    branches = arch_graph.get("branches", [])
    skips = arch_graph.get("skips", [])
    residuals = arch_graph.get("residuals", [])
    src_type = arch_graph.get("primary_topology_source", "TEXT")

    if nodes or edges or tensors or branches or skips or residuals:
        lines.append("## Architecture Graph Topology")
        if src_type == "FIGURE":
            lines.append("Primary Topology Source: FIGURE (Visual Figure Diagram — authoritative source for arrows, branches, skip connections, and residuals)")
        if nodes:
            lines.append("### Nodes")
            for n in nodes:
                n_id = n.get("id") or n.get("name") or "node"
                n_type = n.get("type") or n.get("component_type") or "Unknown"
                lines.append(f"- {n_id}: {n_type}")
        if edges:
            lines.append("### Edges")
            for e in edges:
                from_n = e.get("from") or e.get("from_node") or e.get("source") or "?"
                to_n = e.get("to") or e.get("to_node") or e.get("target") or "?"
                lines.append(f"- {from_n} -> {to_n}")
        if tensors:
            lines.append("### Tensors")
            for t in tensors:
                t_name = t.get("name") or t.get("id") or "tensor"
                lines.append(f"- {t_name}")
        if branches:
            lines.append("### Branches")
            for b in branches:
                lines.append(f"- {b}")
        if skips:
            lines.append("### Skip Connections")
            for s in skips:
                lines.append(f"- {s}")
        if residuals:
            lines.append("### Residual Connections")
            for r in residuals:
                lines.append(f"- {r}")
        lines.append("")

    fps = data.get("forward_pass") or arch_graph.get("forward_pass") or []
    if fps:
        lines.append("## Forward Pass Contract (Execution Graph)")
        lines.append("Code generation MUST follow this exact forward pass execution order instead of guessing:")
        for step in fps:
            if isinstance(step, dict):
                s_num = step.get("step", 1)
                op = step.get("operation", "layer")
                inp = step.get("input", "x")
                out = step.get("output", "out")
                cons = step.get("consumer_operation")
            else:
                s_num = getattr(step, "step", 1)
                op = getattr(step, "operation", "layer")
                inp = getattr(step, "input", "x")
                out = getattr(step, "output", "out")
                cons = getattr(step, "consumer_operation", None)
            cons_str = f" -> Consumer: {cons}" if cons else ""
            cons_json = f', "consumer_operation": "{cons}"' if cons else ""
            lines.append(f"- Step {s_num}: {out} = self.{op}({inp})  (Input: {inp} -> Operation: {op} -> Output: {out}{cons_str})  [JSON: {{\"step\": {s_num}, \"operation\": \"{op}\", \"input\": \"{inp}\", \"output\": \"{out}\"{cons_json}}}]")
        lines.append("")

    tf = data.get("tensor_flow")
    if not tf and isinstance(arch_graph, dict):
        tf = arch_graph.get("tensor_flow")
    elif not tf and hasattr(arch_graph, "get_tensor_flow_chain"):
        tf = arch_graph.get_tensor_flow_chain()
    if tf:
        lines.append("## Preserved Tensor Flow (Architecture)")
        lines.append("The tensor flow IS the architecture. Do NOT merely list static components.")
        lines.append(f"Directed tensor flow chain:\n{tf}")
        lines.append("")

    eqs = data.get("equations", [])
    if eqs:
        lines.append("## Executable Equation Operations")
        lines.append("Code generation MUST implement these deterministic mathematical formulas for model operations:")
        for eq in eqs:
            if isinstance(eq, dict):
                op = eq.get("operation", "operation")
                formula = eq.get("formula", "")
            else:
                op = getattr(eq, "operation", "operation")
                formula = getattr(eq, "formula", str(eq))
            lines.append(f"- operation: {op}")
            lines.append(f"  formula: {formula}")
            lines.append(f"  [JSON: {{\"operation\": \"{op}\", \"formula\": \"{formula}\"}}]")
        lines.append("")

    is_architecture_only = bool(target_file and (target_file.endswith("models.py") or target_file.endswith("test_model.py")))
    if is_architecture_only:
        return "\n".join(lines)

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
            lines.append(f"- {_fact_req_prefix(fact)}{f_name}{val_str}{prov_str}")
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
            lines.append(f"- {_fact_req_prefix(fact)}{f_name}{val_str}{prov_str}")
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
