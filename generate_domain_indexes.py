#!/usr/bin/env python3
"""
Generate domain-specific paper index reports from the deduped v3 graph.
Also splits the General domain into Reconstruction / Rendering / Enhancement / Evaluation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2" / "final_reviewed_v3" / "deduped"
INPUT_JSON = BASE_DIR / "graphify_final_reviewed_v3_deduped.json"
OUT_DIR = BASE_DIR / "indexes"
GENERAL_SUBINDEX_DIR = OUT_DIR / "general_subindexes"

DOMAINS = ["SLAM", "Robotics", "General"]
GENERAL_SUBDOMAINS = ["Reconstruction", "Rendering", "Enhancement", "Evaluation"]


def load_graph() -> dict:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERAL_SUBINDEX_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_by_id = {node["id"]: node for node in nodes}

    doc_edges = defaultdict(list)
    category_docs = defaultdict(list)
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        rel = edge.get("relation")
        if src in node_by_id and tgt in node_by_id and node_by_id[src].get("file_type") in {"document", "paper"}:
            doc_edges[src].append((tgt, rel))
            if rel == "belongs_to":
                category_docs[node_by_id[tgt].get("label", "Unknown")].append(src)

    for domain in DOMAINS:
        docs = sorted(set(category_docs.get(domain, [])), key=lambda x: node_by_id[x].get("label", ""))
        datasets = Counter()
        metrics = Counter()
        concepts = Counter()
        paper_rows = []
        for doc_id in docs:
            node = node_by_id[doc_id]
            for target, rel in doc_edges.get(doc_id, []):
                target_node = node_by_id.get(target, {})
                if rel == "belongs_to":
                    continue
                target_id = target_node.get("id", "")
                target_label = target_node.get("label", "")
                if target_id.startswith("dataset_") or target_node.get("file_type") == "dataset":
                    datasets[target_label] += 1
                elif target_id.startswith("metric_") or target_node.get("file_type") == "metric":
                    metrics[target_label] += 1
                elif target_node.get("file_type") in {"concept", "method"}:
                    concepts[target_label] += 1
            paper_rows.append(
                {
                    "id": doc_id,
                    "label": node.get("label", ""),
                    "source_file": node.get("source_file", ""),
                    "alias_count": len(node.get("alias_ids", [])),
                }
            )

        payload = {
            "domain": domain,
            "paper_count": len(paper_rows),
            "top_datasets": counter_rows(datasets, 20),
            "top_metrics": counter_rows(metrics, 20),
            "top_concepts": counter_rows(concepts, 20),
            "papers": paper_rows,
        }
        (OUT_DIR / f"{domain.lower()}_index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUT_DIR / f"{domain.lower()}_index.md").write_text(render_markdown(payload), encoding="utf-8")

        if domain == "General":
            generate_general_subindexes(node_by_id, doc_edges, paper_rows)

    print(json.dumps({"domains": DOMAINS}, ensure_ascii=False))


def counter_rows(counter: Counter, limit: int) -> list[dict]:
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def render_markdown(payload: dict) -> str:
    title = f"{payload['domain']} Index"
    if payload.get("subdomain"):
        title = f"{payload['domain']} - {payload['subdomain']} Index"
    lines = [f"# {title}", "", f"- paper count: {payload['paper_count']}", ""]
    lines.append("## Top Datasets")
    lines.append("")
    for row in payload["top_datasets"]:
        lines.append(f"- {row['label']}: {row['count']}")
    lines.extend(["", "## Top Metrics", ""])
    for row in payload["top_metrics"]:
        lines.append(f"- {row['label']}: {row['count']}")
    lines.extend(["", "## Top Concepts", ""])
    for row in payload["top_concepts"]:
        lines.append(f"- {row['label']}: {row['count']}")
    lines.extend(["", "## Papers", ""])
    for row in payload["papers"]:
        alias_note = f" | aliases {row['alias_count']}" if row["alias_count"] else ""
        lines.append(f"- {row['label']} | {row['source_file']}{alias_note}")
    lines.append("")
    return "\n".join(lines)


def generate_general_subindexes(node_by_id: dict, doc_edges: dict, paper_rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in paper_rows:
        subdomain = infer_general_subdomain(row["label"], row["source_file"])
        grouped[subdomain].append(row)

    summary = {}
    for subdomain in GENERAL_SUBDOMAINS:
        docs = sorted(grouped.get(subdomain, []), key=lambda x: x["label"])
        datasets = Counter()
        metrics = Counter()
        concepts = Counter()
        for row in docs:
            for target, rel in doc_edges.get(row["id"], []):
                target_node = node_by_id.get(target, {})
                if rel == "belongs_to":
                    continue
                target_id = target_node.get("id", "")
                target_label = target_node.get("label", "")
                if target_id.startswith("dataset_") or target_node.get("file_type") == "dataset":
                    datasets[target_label] += 1
                elif target_id.startswith("metric_") or target_node.get("file_type") == "metric":
                    metrics[target_label] += 1
                elif target_node.get("file_type") in {"concept", "method"}:
                    concepts[target_label] += 1

        payload = {
            "domain": "General",
            "subdomain": subdomain,
            "paper_count": len(docs),
            "top_datasets": counter_rows(datasets, 20),
            "top_metrics": counter_rows(metrics, 20),
            "top_concepts": counter_rows(concepts, 20),
            "papers": docs,
        }
        stem = subdomain.lower()
        (GENERAL_SUBINDEX_DIR / f"{stem}_index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (GENERAL_SUBINDEX_DIR / f"{stem}_index.md").write_text(
            render_markdown(payload),
            encoding="utf-8",
        )
        summary[subdomain] = len(docs)

    (GENERAL_SUBINDEX_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_general_subdomain(label: str, source_file: str) -> str:
    text = f"{label} {source_file}".lower()

    evaluation_terms = [
        "benchmark", "evaluation", "eval", "metric", "survey", "analysis",
        "compare", "comparison", "watermark", "adversarial", "locating",
    ]
    enhancement_terms = [
        "enhance", "enhancement", "denoise", "deblur", "harmonization",
        "inpaint", "editing", "edit", "compression", "stream", "streaming",
        "super-resolution", "repair", "restore", "refinement", "makeup",
    ]
    rendering_terms = [
        "render", "relight", "relighting", "synthesis", "avatar", "talking head",
        "text-to-3d", "video", "dynamic", "view synthesis", "novel view",
        "animation", "generat", "scene generation", "driving",
    ]
    reconstruction_terms = [
        "reconstruction", "reconstruct", "mvs", "stereo", "mapping", "surface",
        "mesh", "geometry", "point map", "lidar", "panoramic", "calibration",
        "segmentation", "scene", "colmap", "pose", "splatting",
    ]

    if any(term in text for term in evaluation_terms):
        return "Evaluation"
    if any(term in text for term in enhancement_terms):
        return "Enhancement"
    if any(term in text for term in rendering_terms):
        return "Rendering"
    if any(term in text for term in reconstruction_terms):
        return "Reconstruction"
    return "Reconstruction"


if __name__ == "__main__":
    main()
