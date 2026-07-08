import re
from typing import Any
from adapters.knowledge_adapter import KnowledgeAdapter
from adapters.research_adapter import ResearchAdapter


class EvidenceRouter:
    """
    Intelligent router that evaluates when to query ForgeKnowledge
    and ForgeResearch before or after execution.
    """

    def __init__(
        self,
        knowledge: KnowledgeAdapter | None = None,
        research: ResearchAdapter | None = None,
    ):
        self.knowledge = knowledge
        self.research = research

    def route_pre_execution(
        self,
        spec: str,
    ) -> dict[str, Any]:
        """
        Analyze spec before planning/generation.
        Returns retrieved knowledge_evidence and research_evidence.
        """
        spec_lower = spec.lower()
        knowledge_evidence = []
        research_evidence = []

        # 1. Check if we need ForgeKnowledge (Official API Truth)
        api_keywords = [
            "convtranspose", "adaptiveavgpool", "batchnorm", "layernorm",
            "transformer", "attention", "embedding", "conv1d", "conv2d",
            "conv3d", "maxpool", "avgpool", "linear", "relu", "gelu",
            "silu", "dropout", "output_padding", "stride", "dilation"
        ]
        needs_api = any(kw in spec_lower for kw in api_keywords)

        if needs_api and self.knowledge:
            # Try searching docs or exact symbol
            for token in re.findall(r'[a-zA-Z0-9_.]+', spec):
                if any(token.lower().startswith(kw) for kw in api_keywords) and len(token) > 3:
                    symbol_doc = (
                        self.knowledge.get_symbol(token)
                        if "." in token or token[0].isupper()
                        else None
                    )
                    if symbol_doc:
                        knowledge_evidence.append(symbol_doc)
                    else:
                        search_res = self.knowledge.search(token, limit=2)
                        if search_res:
                            knowledge_evidence.extend(search_res)

            # If no specific tokens matched well, run a general search on key terms
            if not knowledge_evidence:
                search_res = self.knowledge.search(spec[:100], limit=3)
                if search_res:
                    knowledge_evidence.extend(search_res)

        # Remove duplicates by symbol or title
        unique_knowledge = {}
        for item in knowledge_evidence:
            key = item.get("symbol") or item.get("title") or str(item)
            unique_knowledge[key] = item
        knowledge_evidence = list(unique_knowledge.values())[:3]

        # 2. Check if we need ForgeResearch (External research evidence)
        research_keywords = [
            "paper", "transformer", "resnet", "attention is all you need",
            "arxiv", "et al", "proposed by", "authors", "repository",
            "github", "supplementary", "dataset", "original implementation"
        ]
        needs_research = any(kw in spec_lower for kw in research_keywords)

        if needs_research and self.research:
            lines = [l.strip() for l in spec.split("\n") if l.strip()]
            query_title = lines[0] if lines else spec[:80]
            # Strip markdown heading and "Project:" prefix to get the actual paper title
            query_title = re.sub(r'^#+\s*', '', query_title)
            query_title = re.sub(r'^Project:\s*', '', query_title, flags=re.IGNORECASE)
            query_title = query_title.strip() or spec[:80]
            try:
                papers = self.research.find_papers(title=query_title, limit=2)
                for p in papers:
                    research_evidence.append({
                        "paper": p,
                        "title": p.get("title"),
                        "authors": p.get("authors", []),
                        "landing_page": p.get("urls", {}).get("research_landing_page"),
                        "pdf_url": p.get("urls", {}).get("open_access_pdf")
                    })
            except Exception as e:
                research_evidence.append({"error": f"Research query failed: {e}"})

        return {
            "knowledge_evidence": knowledge_evidence,
            "research_evidence": research_evidence,
            "routed_to_knowledge": len(knowledge_evidence) > 0,
            "routed_to_research": len(research_evidence) > 0,
        }

    def route_post_failure(
        self,
        failed_stage: str,
        error_output: str,
    ) -> dict[str, Any]:
        """
        Analyze execution/syntax failure to diagnose if API truth or research info is needed.
        """
        error_lower = error_output.lower()
        diagnostic_knowledge = []
        diagnostic_research = []

        if any(
            err_kw in error_lower
            for err_kw in [
                "attributeerror", "typeerror", "valueerror",
                "unexpected keyword argument", "got an unexpected",
                "missing required", "no attribute"
            ]
        ):
            if self.knowledge:
                tokens = re.findall(r"'([^']+)'", error_output)
                for t in tokens:
                    if len(t) > 2:
                        res = self.knowledge.search(t, limit=2)
                        if res:
                            diagnostic_knowledge.extend(res)
                if not diagnostic_knowledge:
                    res = self.knowledge.search(error_output[:100], limit=2)
                    if res:
                        diagnostic_knowledge.extend(res)

        if any(
            err_kw in error_lower
            for err_kw in [
                "assert", "shape", "dimension", "size mismatch",
                "expected shape"
            ]
        ):
            if self.research:
                try:
                    papers = self.research.find_papers(
                        title="1D CNN classifier architecture dimensions",
                        limit=1
                    )
                    if papers:
                        diagnostic_research.extend(papers)
                except Exception:
                    pass

        return {
            "diagnostic_knowledge": diagnostic_knowledge,
            "diagnostic_research": diagnostic_research,
            "routed_to_knowledge": len(diagnostic_knowledge) > 0,
            "routed_to_research": len(diagnostic_research) > 0,
        }


def enrich_spec_with_evidence(
    spec: str,
    evidence: dict[str, Any],
) -> str:
    """
    Format spec with retrieved knowledge and research evidence so ForgeCode
    receives grounded context without breaking any existing signatures.
    """
    if (
        not evidence.get("routed_to_knowledge")
        and not evidence.get("routed_to_research")
    ):
        return spec

    sections = [
        spec.strip(),
        "\n\n---",
        "# GROUNDED EVIDENCE BUNDLE (Retrieved by ForgeCore Router)"
    ]

    if evidence.get("routed_to_knowledge"):
        sections.append("\n## Official API Truth (ForgeKnowledge):")
        for item in evidence["knowledge_evidence"]:
            symbol = item.get("symbol") or item.get("title") or "API Reference"
            sig = item.get("signature") or ""
            desc = item.get("description") or ""
            params = item.get("parameters") or {}
            param_str = "\n".join(
                f"  - {k}: {v}"
                for k, v in params.items()
            )
            sections.append(
                f"### {symbol}\nSignature: `{sig}`\nDescription: {desc}"
            )
            if param_str:
                sections.append(f"Parameters:\n{param_str}")

    if evidence.get("routed_to_research"):
        sections.append("\n## Research Evidence (ForgeResearch):")
        for item in evidence["research_evidence"]:
            title = item.get("title") or "Research Paper"
            authors = ", ".join(item.get("authors", [])[:3])
            landing = item.get("landing_page") or "N/A"
            pdf = item.get("pdf_url") or "N/A"
            sections.append(
                f"### {title}\nAuthors: {authors}\n"
                f"Landing Page: {landing}\nOpen Access PDF: {pdf}"
            )

    return "\n".join(sections)
