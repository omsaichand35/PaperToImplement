"""
Null Entity Resolution Engine for PaperForge.

Resolves null/UNKNOWN values in ImplementationSpec using a 3-tier cascade:
  Tier 1: Canonical Framework Registries (torchvision, etc.)
  Tier 2: Consensus Literature Search (ForgeResearch / OpenAlex)
  Tier 3: Domain Default Heuristics (safe ML engineering defaults)
"""
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tier 1: Canonical Dataset Registry
# ---------------------------------------------------------------------------
CANONICAL_DATASETS: dict[str, dict[str, Any]] = {
    "imagenet": {
        "loader": "torchvision.datasets.ImageNet(root=data_path, split='train')",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 1000,
        "input_size": 224,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.ImageNet",
    },
    "cifar-10": {
        "loader": "torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)",
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616],
        "num_classes": 10,
        "input_size": 32,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.CIFAR10",
    },
    "cifar-100": {
        "loader": "torchvision.datasets.CIFAR100(root=data_path, train=True, download=True)",
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
        "num_classes": 100,
        "input_size": 32,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.CIFAR100",
    },
    "mnist": {
        "loader": "torchvision.datasets.MNIST(root=data_path, train=True, download=True)",
        "mean": [0.1307],
        "std": [0.3081],
        "num_classes": 10,
        "input_size": 28,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.MNIST",
    },
    "fashionmnist": {
        "loader": "torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True)",
        "mean": [0.2860],
        "std": [0.3530],
        "num_classes": 10,
        "input_size": 28,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.FashionMNIST",
    },
    "stl-10": {
        "loader": "torchvision.datasets.STL10(root=data_path, split='train', download=True)",
        "mean": [0.4467, 0.4398, 0.4066],
        "std": [0.2242, 0.2215, 0.2239],
        "num_classes": 10,
        "input_size": 96,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.STL10",
    },
    "svhn": {
        "loader": "torchvision.datasets.SVHN(root=data_path, split='train', download=True)",
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
        "num_classes": 10,
        "input_size": 32,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.SVHN",
    },
    "caltech-101": {
        "loader": "torchvision.datasets.Caltech101(root=data_path, download=True)",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 101,
        "input_size": 224,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.Caltech101",
    },
    "caltech-256": {
        "loader": "torchvision.datasets.Caltech256(root=data_path, download=True)",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 257,
        "input_size": 224,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.Caltech256",
    },
    "voc": {
        "loader": "torchvision.datasets.VOCDetection(root=data_path, year='2012', image_set='train', download=True)",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 20,
        "input_size": 224,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.VOCDetection",
    },
    "coco": {
        "loader": "torchvision.datasets.CocoDetection(root=data_path, annFile=ann_path)",
        "mean": [0.471, 0.448, 0.408],
        "std": [0.234, 0.239, 0.242],
        "num_classes": 80,
        "input_size": 224,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.CocoDetection",
    },
    "places365": {
        "loader": "torchvision.datasets.Places365(root=data_path, split='train-standard', download=True)",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 365,
        "input_size": 224,
        "loss": "nn.CrossEntropyLoss()",
        "source": "torchvision.datasets.Places365",
    },
}

# Aliases for flexible matching
_DATASET_ALIASES: dict[str, str] = {
    "imagenet-1k": "imagenet",
    "ilsvrc": "imagenet",
    "ilsvrc2012": "imagenet",
    "ilsvrc-2012": "imagenet",
    "cifar10": "cifar-10",
    "cifar100": "cifar-100",
    "fashion-mnist": "fashionmnist",
    "fashion_mnist": "fashionmnist",
    "stl10": "stl-10",
    "caltech101": "caltech-101",
    "caltech256": "caltech-256",
    "pascalvoc": "voc",
    "pascal_voc": "voc",
    "pascal-voc": "voc",
    "mscoco": "coco",
    "ms-coco": "coco",
}

# ---------------------------------------------------------------------------
# Tier 3: Task-Based Domain Defaults
# ---------------------------------------------------------------------------
TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "classification": {
        "loss": "nn.CrossEntropyLoss()",
        "optimizer": "optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)",
        "epochs": 90,
        "scheduler": "optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)",
    },
    "detection": {
        "loss": "nn.CrossEntropyLoss() + nn.SmoothL1Loss()",
        "optimizer": "optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)",
        "epochs": 50,
        "scheduler": "optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)",
    },
    "segmentation": {
        "loss": "nn.CrossEntropyLoss()",
        "optimizer": "optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)",
        "epochs": 60,
        "scheduler": "optim.lr_scheduler.PolynomialLR(optimizer, total_iters=60, power=0.9)",
    },
}


# ---------------------------------------------------------------------------
# Stage 1: Gap Scanner
# ---------------------------------------------------------------------------

def _normalize_dataset_key(name: str) -> str | None:
    """Normalize a dataset name to a canonical registry key."""
    key = name.lower().strip().replace(" ", "").replace("_", "-")
    if key in CANONICAL_DATASETS:
        return key
    if key in _DATASET_ALIASES:
        return _DATASET_ALIASES[key]
    # Fuzzy: check if any canonical key is a substring
    for canon_key in CANONICAL_DATASETS:
        if canon_key in key or key in canon_key:
            return canon_key
    return None


def detect_dataset_name(
    spec_data: dict,
    user_dataset: str | None = None,
) -> str | None:
    """
    Detect the dataset name from user input, task field,
    paper title, or evidence quotes.
    """
    # Priority 1: Explicit user input
    if user_dataset:
        resolved = _normalize_dataset_key(user_dataset)
        if resolved:
            return resolved
        # Return raw user input even if not in registry (for Tier 2)
        return user_dataset.strip().lower()

    # Priority 2: Scan spec fields
    search_text = " ".join([
        spec_data.get("paper_title") or "",
        spec_data.get("task") or "",
    ]).lower()

    # Also scan evidence quotes in preprocessing
    for fact in spec_data.get("preprocessing", []):
        for ev in fact.get("evidence", []):
            quote = ev.get("quote") or ""
            search_text += " " + quote.lower()

    # Check each canonical dataset name
    for canon_key in CANONICAL_DATASETS:
        # Look for the dataset name as a whole word
        pattern = r'\b' + re.escape(canon_key.replace("-", r"[\s\-]?")) + r'\b'
        if re.search(pattern, search_text):
            return canon_key

    # Check aliases
    for alias, canon_key in _DATASET_ALIASES.items():
        clean_alias = alias.replace("-", " ").replace("_", " ")
        if clean_alias in search_text:
            return canon_key

    return None


def _infer_task_type(spec_data: dict) -> str:
    """Infer the ML task type from spec metadata."""
    search_text = " ".join([
        spec_data.get("task") or "",
        spec_data.get("paper_title") or "",
    ]).lower()

    if any(kw in search_text for kw in ["detection", "yolo", "ssd", "rcnn", "faster"]):
        return "detection"
    if any(kw in search_text for kw in ["segment", "semantic", "unet", "mask"]):
        return "segmentation"
    # Default to classification for CNN / image tasks
    return "classification"


# Gap categories with the fact names we look for
_DATASET_GAP_NAMES = {
    "dataset loader", "data loader", "dataset loading",
    "normalization", "normalization statistics", "mean subtraction",
    "input size", "input resolution", "input shapes", "input_size",
    "num_classes", "number of classes",
}

_OPTIMIZATION_GAP_NAMES = {
    "loss function", "loss functions", "loss",
    "optimizer", "optimiser",
    "epochs", "number of epochs", "total epochs",
    "scheduler", "schedulers", "learning rate schedule",
    "learning rate scheduler", "lr scheduler",
}

_PREPROCESSING_GAP_NAMES = {
    "preprocessing", "image resizing", "data augmentation",
    "input shapes", "output shapes",
}


def scan_gaps(spec_data: dict) -> dict[str, list[dict]]:
    """
    Scan the spec for null/UNKNOWN facts and categorize them.
    Returns dict with keys: dataset_gaps, optimization_gaps, methodology_gaps.
    """
    dataset_gaps: list[dict] = []
    optimization_gaps: list[dict] = []
    methodology_gaps: list[dict] = []

    # Scan unknowns section
    for fact in spec_data.get("unknowns", []):
        name_lower = (fact.get("name") or "").lower().strip()
        if fact.get("value") is not None and fact.get("status") != "UNKNOWN":
            continue
        if any(kw in name_lower for kw in _DATASET_GAP_NAMES):
            dataset_gaps.append(fact)
        elif any(kw in name_lower for kw in _OPTIMIZATION_GAP_NAMES):
            optimization_gaps.append(fact)
        elif any(kw in name_lower for kw in _PREPROCESSING_GAP_NAMES):
            methodology_gaps.append(fact)
        else:
            # Try to categorize by broader matching
            if any(kw in name_lower for kw in ["dataset", "loader", "data", "mean", "std"]):
                dataset_gaps.append(fact)
            elif any(kw in name_lower for kw in ["optim", "loss", "epoch", "lr", "learn"]):
                optimization_gaps.append(fact)
            else:
                methodology_gaps.append(fact)

    # Also check training section for null-valued facts
    for fact in spec_data.get("training", []):
        if fact.get("value") is None or fact.get("status") == "UNKNOWN":
            name_lower = (fact.get("name") or "").lower().strip()
            optimization_gaps.append(fact)

    # Check preprocessing for null-valued facts
    for fact in spec_data.get("preprocessing", []):
        if fact.get("value") is None or fact.get("status") == "UNKNOWN":
            methodology_gaps.append(fact)

    return {
        "dataset_gaps": dataset_gaps,
        "optimization_gaps": optimization_gaps,
        "methodology_gaps": methodology_gaps,
    }


# ---------------------------------------------------------------------------
# Stage 2: Tier 1 — Canonical Framework Registry Resolution
# ---------------------------------------------------------------------------

def _make_fact(name: str, value: Any, status: str, confidence: float, source: str) -> dict:
    """Create a properly structured ImplementationFact dict."""
    return {
        "name": name,
        "value": value,
        "status": status,
        "confidence": confidence,
        "evidence": [{"page": None, "section": None, "quote": f"Source: {source}"}],
        "notes": f"Resolved by Null Entity Resolution Engine ({status})",
    }


def resolve_via_registry(
    gaps: dict[str, list[dict]],
    dataset_key: str | None,
) -> dict[str, list[dict]]:
    """
    Tier 1: Look up canonical framework registries for dataset-specific values.
    Returns dict of new facts organized by target section (training, preprocessing).
    """
    resolved: dict[str, list[dict]] = {"training": [], "preprocessing": []}

    if not dataset_key or dataset_key not in CANONICAL_DATASETS:
        return resolved

    registry = CANONICAL_DATASETS[dataset_key]
    source = registry["source"]

    # Track which gap names have been resolved
    resolved_names: set[str] = set()

    # Resolve dataset gaps
    for gap in gaps.get("dataset_gaps", []):
        name_lower = (gap.get("name") or "").lower()

        if any(kw in name_lower for kw in ["normalization", "mean"]) and "mean" in registry:
            resolved["preprocessing"].append(_make_fact(
                name="normalization_mean",
                value=registry["mean"],
                status="REGISTRY_CANONICAL",
                confidence=1.0,
                source=source,
            ))
            resolved["preprocessing"].append(_make_fact(
                name="normalization_std",
                value=registry["std"],
                status="REGISTRY_CANONICAL",
                confidence=1.0,
                source=source,
            ))
            resolved_names.add(name_lower)

        elif any(kw in name_lower for kw in ["input", "resolution", "size"]) and "input_size" in registry:
            resolved["preprocessing"].append(_make_fact(
                name="input_size",
                value=registry["input_size"],
                status="REGISTRY_CANONICAL",
                confidence=1.0,
                source=source,
            ))
            resolved_names.add(name_lower)

        elif any(kw in name_lower for kw in ["class", "num_class"]) and "num_classes" in registry:
            resolved["training"].append(_make_fact(
                name="num_classes",
                value=registry["num_classes"],
                status="REGISTRY_CANONICAL",
                confidence=1.0,
                source=source,
            ))
            resolved_names.add(name_lower)

        elif any(kw in name_lower for kw in ["loader", "loading", "dataset"]) and "loader" in registry:
            resolved["preprocessing"].append(_make_fact(
                name="dataset_loader",
                value=registry["loader"],
                status="REGISTRY_CANONICAL",
                confidence=1.0,
                source=source,
            ))
            resolved_names.add(name_lower)

    # Resolve optimization gaps using registry loss
    for gap in gaps.get("optimization_gaps", []):
        name_lower = (gap.get("name") or "").lower()
        if any(kw in name_lower for kw in ["loss"]) and "loss" in registry:
            resolved["training"].append(_make_fact(
                name="loss_function",
                value=registry["loss"],
                status="REGISTRY_CANONICAL",
                confidence=1.0,
                source=source,
            ))
            resolved_names.add(name_lower)

    return resolved


# ---------------------------------------------------------------------------
# Stage 2: Tier 2 — Consensus Literature Search
# ---------------------------------------------------------------------------

def resolve_via_literature(
    gaps: dict[str, list[dict]],
    dataset_name: str | None,
    task_type: str,
    research_adapter: Any | None,
) -> dict[str, list[dict]]:
    """
    Tier 2: Query ForgeResearch / OpenAlex for consensus baseline methodologies.
    Uses a keyword voting system across retrieved abstracts.
    """
    resolved: dict[str, list[dict]] = {"training": [], "preprocessing": []}

    if not research_adapter:
        return resolved

    remaining_opt_gaps = [
        g for g in gaps.get("optimization_gaps", [])
        if g.get("value") is None or g.get("status") == "UNKNOWN"
    ]

    if not remaining_opt_gaps:
        return resolved

    # Build targeted query
    ds_part = f" {dataset_name}" if dataset_name else ""
    query = f"baseline {task_type}{ds_part} training optimizer loss function methodology"

    try:
        papers = research_adapter.find_papers(title=query, limit=5)
    except Exception:
        return resolved

    if not papers:
        return resolved

    # Collect all titles for keyword voting
    all_text = " ".join(
        (p.get("title") or "") for p in papers
    ).lower()

    # Vote on optimizer
    optimizer_votes = {
        "sgd": all_text.count("sgd") + all_text.count("stochastic gradient"),
        "adam": all_text.count("adam"),
        "adamw": all_text.count("adamw"),
    }
    best_optim = max(optimizer_votes, key=optimizer_votes.get)  # type: ignore[arg-type]

    if optimizer_votes[best_optim] > 0:
        optim_map = {
            "sgd": "optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)",
            "adam": "optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)",
            "adamw": "optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)",
        }
        doi_list = [p.get("doi") or "N/A" for p in papers[:3]]
        source_str = f"Literature consensus from {len(papers)} papers (DOIs: {', '.join(doi_list)})"

        for gap in remaining_opt_gaps:
            name_lower = (gap.get("name") or "").lower()
            if "optim" in name_lower:
                resolved["training"].append(_make_fact(
                    name="optimizer",
                    value=optim_map[best_optim],
                    status="LITERATURE_GROUNDED",
                    confidence=0.85,
                    source=source_str,
                ))

    # Vote on loss function
    loss_votes = {
        "cross_entropy": all_text.count("cross") + all_text.count("entropy"),
        "mse": all_text.count("mse") + all_text.count("mean squared"),
        "bce": all_text.count("bce") + all_text.count("binary cross"),
    }
    best_loss = max(loss_votes, key=loss_votes.get)  # type: ignore[arg-type]

    if loss_votes[best_loss] > 0:
        loss_map = {
            "cross_entropy": "nn.CrossEntropyLoss()",
            "mse": "nn.MSELoss()",
            "bce": "nn.BCEWithLogitsLoss()",
        }
        for gap in remaining_opt_gaps:
            name_lower = (gap.get("name") or "").lower()
            if "loss" in name_lower:
                resolved["training"].append(_make_fact(
                    name="loss_function",
                    value=loss_map[best_loss],
                    status="LITERATURE_GROUNDED",
                    confidence=0.85,
                    source=f"Literature consensus from {len(papers)} papers",
                ))

    return resolved


# ---------------------------------------------------------------------------
# Stage 2: Tier 3 — Domain Default Heuristics
# ---------------------------------------------------------------------------

def resolve_via_heuristics(
    gaps: dict[str, list[dict]],
    task_type: str,
) -> dict[str, list[dict]]:
    """
    Tier 3: Apply safe ML engineering defaults for any remaining gaps.
    """
    resolved: dict[str, list[dict]] = {"training": [], "preprocessing": []}
    defaults = TASK_DEFAULTS.get(task_type, TASK_DEFAULTS["classification"])

    resolved_names: set[str] = set()

    for gap in gaps.get("optimization_gaps", []):
        name_lower = (gap.get("name") or "").lower()

        if any(kw in name_lower for kw in ["loss"]) and "loss" not in resolved_names:
            resolved["training"].append(_make_fact(
                name="loss_function",
                value=defaults["loss"],
                status="DOMAIN_HEURISTIC",
                confidence=0.6,
                source=f"Domain default for {task_type}",
            ))
            resolved_names.add("loss")

        elif any(kw in name_lower for kw in ["optim"]) and "optimizer" not in resolved_names:
            resolved["training"].append(_make_fact(
                name="optimizer",
                value=defaults["optimizer"],
                status="DOMAIN_HEURISTIC",
                confidence=0.6,
                source=f"Domain default for {task_type}",
            ))
            resolved_names.add("optimizer")

        elif any(kw in name_lower for kw in ["epoch"]) and "epochs" not in resolved_names:
            resolved["training"].append(_make_fact(
                name="epochs",
                value=defaults["epochs"],
                status="DOMAIN_HEURISTIC",
                confidence=0.6,
                source=f"Domain default for {task_type}",
            ))
            resolved_names.add("epochs")

        elif any(kw in name_lower for kw in ["schedul", "lr"]) and "scheduler" not in resolved_names:
            resolved["training"].append(_make_fact(
                name="scheduler",
                value=defaults["scheduler"],
                status="DOMAIN_HEURISTIC",
                confidence=0.6,
                source=f"Domain default for {task_type}",
            ))
            resolved_names.add("scheduler")

    return resolved


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def resolve_null_entities(
    spec_obj: Any,
    dataset_name: str | None = None,
    research: Any | None = None,
) -> dict[str, Any]:
    """
    Main orchestrator: runs Tier 1 -> 2 -> 3 resolution cascade.
    Mutates spec_obj in place and returns a resolution report.
    """
    # Convert to dict for manipulation
    if hasattr(spec_obj, "model_dump"):
        spec_data = spec_obj.model_dump()
    elif isinstance(spec_obj, dict):
        spec_data = spec_obj
    else:
        return {"resolved_count": 0, "tiers": {}}

    report: dict[str, Any] = {
        "resolved_count": 0,
        "tiers": {"registry": 0, "literature": 0, "heuristic": 0},
        "details": [],
    }

    # Detect dataset
    detected_dataset = detect_dataset_name(spec_data, dataset_name)
    report["detected_dataset"] = detected_dataset

    # Infer task type
    task_type = _infer_task_type(spec_data)
    report["inferred_task"] = task_type

    # Scan all gaps
    gaps = scan_gaps(spec_data)
    total_gaps = sum(len(v) for v in gaps.values())
    report["total_gaps_found"] = total_gaps
    print(f"[NullResolver] Detected dataset: {detected_dataset or 'unknown'}, task: {task_type}, gaps: {total_gaps}", flush=True)

    if total_gaps == 0:
        print("[NullResolver] No gaps to resolve.", flush=True)
        return report

    # Track which gap names are already resolved to avoid duplicates
    already_resolved: set[str] = set()

    # Tier 1: Registry
    print("[NullResolver] Tier 1: Checking canonical framework registries...", flush=True)
    registry_key = _normalize_dataset_key(detected_dataset) if detected_dataset else None
    tier1 = resolve_via_registry(gaps, registry_key)
    for section, facts in tier1.items():
        for fact in facts:
            fact_name = fact["name"].lower()
            if fact_name not in already_resolved:
                spec_data.setdefault(section, []).append(fact)
                already_resolved.add(fact_name)
                report["tiers"]["registry"] += 1
                report["details"].append(f"REGISTRY_CANONICAL: {fact['name']} = {fact['value']}")
    print(f"[NullResolver]   -> Resolved {report['tiers']['registry']} facts via registry.", flush=True)

    # Remove resolved gaps from the working set for subsequent tiers
    _remove_resolved_gaps(gaps, already_resolved)

    # Tier 2: Literature
    remaining = sum(len(v) for v in gaps.values())
    if remaining > 0:
        print(f"[NullResolver] Tier 2: Querying literature for {remaining} remaining gaps...", flush=True)
        tier2 = resolve_via_literature(gaps, detected_dataset, task_type, research)
        for section, facts in tier2.items():
            for fact in facts:
                fact_name = fact["name"].lower()
                if fact_name not in already_resolved:
                    spec_data.setdefault(section, []).append(fact)
                    already_resolved.add(fact_name)
                    report["tiers"]["literature"] += 1
                    report["details"].append(f"LITERATURE_GROUNDED: {fact['name']} = {fact['value']}")
        print(f"[NullResolver]   -> Resolved {report['tiers']['literature']} facts via literature.", flush=True)
        _remove_resolved_gaps(gaps, already_resolved)

    # Tier 3: Heuristics
    remaining = sum(len(v) for v in gaps.values())
    if remaining > 0:
        print(f"[NullResolver] Tier 3: Applying domain heuristics for {remaining} remaining gaps...", flush=True)
        tier3 = resolve_via_heuristics(gaps, task_type)
        for section, facts in tier3.items():
            for fact in facts:
                fact_name = fact["name"].lower()
                if fact_name not in already_resolved:
                    spec_data.setdefault(section, []).append(fact)
                    already_resolved.add(fact_name)
                    report["tiers"]["heuristic"] += 1
                    report["details"].append(f"DOMAIN_HEURISTIC: {fact['name']} = {fact['value']}")
        print(f"[NullResolver]   -> Resolved {report['tiers']['heuristic']} facts via heuristics.", flush=True)

    # Remove resolved items from unknowns
    _prune_resolved_unknowns(spec_data, already_resolved)

    report["resolved_count"] = sum(report["tiers"].values())
    print(
        f"[NullResolver] Resolution complete: {report['resolved_count']} facts resolved "
        f"({report['tiers']['registry']} registry, {report['tiers']['literature']} literature, "
        f"{report['tiers']['heuristic']} heuristic).",
        flush=True,
    )

    # Write back to spec_obj if it has Pydantic model interface
    if hasattr(spec_obj, "__dict__"):
        for key in ("preprocessing", "training", "unknowns"):
            if key in spec_data:
                setattr(spec_obj, key, spec_data[key])

    return report


def _remove_resolved_gaps(
    gaps: dict[str, list[dict]],
    resolved_names: set[str],
) -> None:
    """Remove gaps that have been resolved by a previous tier."""
    for category in gaps:
        gaps[category] = [
            g for g in gaps[category]
            if (g.get("name") or "").lower() not in resolved_names
        ]


def _prune_resolved_unknowns(
    spec_data: dict,
    resolved_names: set[str],
) -> None:
    """Remove unknowns whose fact names overlap with resolved names or model_components."""
    unknowns = spec_data.get("unknowns", [])

    # Build a broader set of resolved keywords for fuzzy matching
    resolved_keywords: set[str] = set()
    for name in resolved_names:
        resolved_keywords.update(name.replace("_", " ").split())

    # Also collect fact names from model_components (these are already extracted)
    for comp in spec_data.get("model_components", []):
        comp_name = (comp.get("name") or "").lower()
        resolved_keywords.update(comp_name.replace("_", " ").split())
        for fact in comp.get("facts", []):
            if fact.get("value") is not None:
                fact_name = (fact.get("name") or "").lower()
                resolved_keywords.update(fact_name.replace("_", " ").split())

    # Also collect from training and preprocessing
    for section_key in ("training", "preprocessing"):
        for fact in spec_data.get(section_key, []):
            if fact.get("value") is not None:
                fact_name = (fact.get("name") or "").lower()
                resolved_keywords.update(fact_name.replace("_", " ").split())

    # Remove generic filler words
    resolved_keywords -= {"in", "of", "the", "a", "an", "this", "chunk", "no", "information", "about"}

    pruned: list[dict] = []
    for unknown in unknowns:
        u_name = (unknown.get("name") or "").lower().strip()
        # Exact match
        if u_name in resolved_names:
            continue
        # Word overlap match — if majority of the unknown's key words are covered
        u_words = set(u_name.replace("_", " ").split())
        u_words -= {"in", "of", "the", "a", "an", "this", "chunk", "no", "information", "about"}
        if u_words and len(u_words & resolved_keywords) >= max(1, len(u_words) // 2):
            continue
        pruned.append(unknown)

    spec_data["unknowns"] = pruned


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Null Entity Resolution Engine")
    parser.add_argument("--spec", required=True, help="Path to implementation_spec.json")
    parser.add_argument("--dataset", default=None, help="Dataset name (e.g., ImageNet, CIFAR-10)")
    parser.add_argument("--output", default=None, help="Output path (defaults to overwriting input)")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"[ERROR] Spec file not found: {spec_path}")
        sys.exit(1)

    with open(spec_path, "r", encoding="utf-8") as f:
        spec_data = json.load(f)

    report = resolve_null_entities(spec_data, dataset_name=args.dataset)

    out_path = Path(args.output) if args.output else spec_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec_data, f, indent=2, ensure_ascii=False)

    print(f"\nEnriched spec saved to: {out_path}")
    print(f"Resolution report: {json.dumps(report, indent=2)}")
