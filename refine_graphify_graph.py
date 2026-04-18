#!/usr/bin/env python3
"""
Refine the merged graph by pruning overly generic high-frequency concepts and
produce a category-clustered report for downstream review.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
GRAPH_DIR = ROOT / "graphify-out"
MERGED_JSON = GRAPH_DIR / "graphify_merged.json"
FILTERED_DIR = GRAPH_DIR / "filtered"

GENERIC_LABELS = {
    "3D Gaussian Splatting",
    "NeRF",
    "SLAM",
    "Reconstruction",
    "Navigation",
    "Visual Localization",
    "Semantic Mapping",
    "ATE",
    "ATE RMSE",
    "SPL",
    "IoU",
    "Accuracy",
    "PSNR",
    "SSIM",
    "LPIPS",
}

PROTECTED_RELATIONS = {"belongs_to"}
GENERIC_FILE_TYPES = {"concept", "metric", "method"}
MAX_DOC_COVERAGE = 0.18


def load_graph() -> dict:
    return json.loads(MERGED_JSON.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])

    node_by_id = {node["id"]: node for node in nodes}
    document_ids = {node["id"] for node in nodes if node.get("file_type") == "document"}
    doc_count = len(document_ids)

    target_doc_sets: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in document_ids and target in node_by_id:
            target_doc_sets[target].add(source)

    removed_targets = set()
    removed_reasons = {}
    for target, docs in target_doc_sets.items():
        node = node_by_id[target]
        label = node.get("label", "")
        file_type = node.get("file_type", "")
        coverage = len(docs) / doc_count if doc_count else 0
        if label in GENERIC_LABELS:
            removed_targets.add(target)
            removed_reasons[target] = {"reason": "generic_label", "coverage": coverage}
        elif file_type in GENERIC_FILE_TYPES and coverage > MAX_DOC_COVERAGE:
            removed_targets.add(target)
            removed_reasons[target] = {"reason": "high_doc_coverage", "coverage": coverage}

    filtered_edges = []
    for edge in edges:
        target = edge.get("target")
        relation = edge.get("relation", "")
        if relation not in PROTECTED_RELATIONS and target in removed_targets:
            continue
        filtered_edges.append(edge)

    kept_node_ids = set()
    for edge in filtered_edges:
        kept_node_ids.add(edge.get("source"))
        kept_node_ids.add(edge.get("target"))
    for hyperedge in hyperedges:
        kept_node_ids.add(hyperedge.get("id"))
        kept_node_ids.update(hyperedge.get("nodes", []))

    filtered_nodes = [node for node in nodes if node.get("id") in kept_node_ids]
    filtered_node_by_id = {node["id"]: node for node in filtered_nodes}

    filtered_hyperedges = []
    for hyperedge in hyperedges:
        members = [m for m in hyperedge.get("nodes", []) if m in filtered_node_by_id]
        if members:
            item = dict(hyperedge)
            item["nodes"] = members
            filtered_hyperedges.append(item)

    filtered_graph = {
        "generated_from": str(MERGED_JSON),
        "source_markdown_count": graph.get("source_markdown_count", 0),
        "node_count_before": len(nodes),
        "edge_count_before": len(edges),
        "node_count_after": len(filtered_nodes),
        "edge_count_after": len(filtered_edges),
        "removed_target_count": len(removed_targets),
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "hyperedges": filtered_hyperedges,
    }
    (FILTERED_DIR / "graphify_filtered.json").write_text(
        json.dumps(filtered_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(
        FILTERED_DIR / "graphify_filtered_nodes.csv",
        [
            {
                "id": node.get("id", ""),
                "label": node.get("label", ""),
                "file_type": node.get("file_type", ""),
                "category": node.get("category", ""),
                "source_file": node.get("source_file", ""),
            }
            for node in filtered_nodes
        ],
        ["id", "label", "file_type", "category", "source_file"],
    )
    write_csv(
        FILTERED_DIR / "graphify_filtered_edges.csv",
        [
            {
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "relation": edge.get("relation", ""),
                "confidence": edge.get("confidence", ""),
                "weight": edge.get("weight", ""),
                "source_file": edge.get("source_file", ""),
            }
            for edge in filtered_edges
        ],
        ["source", "target", "relation", "confidence", "weight", "source_file"],
    )

    removed_rows = []
    for node_id, meta in sorted(removed_reasons.items(), key=lambda item: (-item[1]["coverage"], item[0])):
        node = node_by_id[node_id]
        removed_rows.append(
            {
                "id": node_id,
                "label": node.get("label", ""),
                "file_type": node.get("file_type", ""),
                "coverage": round(meta["coverage"], 4),
                "reason": meta["reason"],
                "source_file": node.get("source_file", ""),
            }
        )
    write_csv(
        FILTERED_DIR / "graphify_removed_targets.csv",
        removed_rows,
        ["id", "label", "file_type", "coverage", "reason", "source_file"],
    )

    cluster_report = build_cluster_report(filtered_nodes, filtered_edges, filtered_hyperedges)
    (FILTERED_DIR / "topic_clusters.json").write_text(
        json.dumps(cluster_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (FILTERED_DIR / "topic_clusters.md").write_text(render_cluster_markdown(cluster_report), encoding="utf-8")

    summary = {
        "source_markdown_count": graph.get("source_markdown_count", 0),
        "node_count_before": len(nodes),
        "edge_count_before": len(edges),
        "node_count_after": len(filtered_nodes),
        "edge_count_after": len(filtered_edges),
        "removed_target_count": len(removed_targets),
        "largest_categories": cluster_report["category_sizes"][:10],
    }
    (FILTERED_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def build_cluster_report(nodes: list[dict], edges: list[dict], hyperedges: list[dict]) -> dict:
    node_by_id = {node["id"]: node for node in nodes}
    document_ids = [node["id"] for node in nodes if node.get("file_type") == "document"]
    doc_degree = Counter()
    relation_counts = Counter()
    category_docs = defaultdict(list)
    doc_neighbors = defaultdict(list)

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation", "")
        relation_counts[relation] += 1
        if source in node_by_id and target in node_by_id:
            doc_neighbors[source].append((target, relation))
            if node_by_id[source].get("file_type") == "document":
                doc_degree[source] += 1
        if relation == "belongs_to" and source in node_by_id:
            category = node_by_id[target].get("label", "Unknown")
            category_docs[category].append(source)

    category_sizes = sorted(
        [{"category": category, "document_count": len(doc_ids)} for category, doc_ids in category_docs.items()],
        key=lambda x: (-x["document_count"], x["category"]),
    )

    categories = []
    for category, doc_ids in sorted(category_docs.items(), key=lambda item: (-len(item[1]), item[0])):
        concept_counts = Counter()
        dataset_counts = Counter()
        metric_counts = Counter()
        representative_docs = []
        for doc_id in doc_ids:
            node = node_by_id[doc_id]
            representative_docs.append(
                {
                    "id": doc_id,
                    "label": node.get("label", ""),
                    "source_file": node.get("source_file", ""),
                    "degree": doc_degree[doc_id],
                }
            )
            for target, relation in doc_neighbors.get(doc_id, []):
                target_node = node_by_id.get(target, {})
                target_label = target_node.get("label", "")
                target_type = target_node.get("file_type", "")
                target_id = target_node.get("id", "")
                if relation == "belongs_to":
                    continue
                if target_id.startswith("dataset_"):
                    dataset_counts[target_label] += 1
                elif target_id.startswith("metric_"):
                    metric_counts[target_label] += 1
                elif target_type in {"dataset"}:
                    dataset_counts[target_label] += 1
                elif target_type in {"metric"}:
                    metric_counts[target_label] += 1
                elif target_type in {"concept", "method"}:
                    concept_counts[target_label] += 1

        representative_docs.sort(key=lambda x: (-x["degree"], x["label"]))
        categories.append(
            {
                "category": category,
                "document_count": len(doc_ids),
                "top_concepts": counter_rows(concept_counts, 12),
                "top_datasets": counter_rows(dataset_counts, 8),
                "top_metrics": counter_rows(metric_counts, 8),
                "representative_documents": representative_docs[:15],
            }
        )

    return {
        "category_sizes": category_sizes,
        "relation_counts": counter_rows(relation_counts, 20),
        "categories": categories,
        "hyperedge_count": len(hyperedges),
    }


def counter_rows(counter: Counter, limit: int) -> list[dict]:
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def render_cluster_markdown(report: dict) -> str:
    lines = []
    lines.append("# Topic Clusters")
    lines.append("")
    lines.append("## Category Sizes")
    lines.append("")
    for row in report.get("category_sizes", []):
        lines.append(f"- {row['category']}: {row['document_count']}")

    lines.append("")
    lines.append("## Relation Counts")
    lines.append("")
    for row in report.get("relation_counts", []):
        lines.append(f"- {row['label']}: {row['count']}")

    for category in report.get("categories", []):
        lines.append("")
        lines.append(f"## {category['category']}")
        lines.append("")
        lines.append(f"- documents: {category['document_count']}")
        lines.append("- top concepts: " + join_rows(category["top_concepts"]))
        lines.append("- top datasets: " + join_rows(category["top_datasets"]))
        lines.append("- top metrics: " + join_rows(category["top_metrics"]))
        lines.append("- representative documents:")
        for doc in category["representative_documents"]:
            lines.append(f"  - {doc['label']} ({doc['degree']})")

    lines.append("")
    lines.append(f"## Hyperedges")
    lines.append("")
    lines.append(f"- count: {report.get('hyperedge_count', 0)}")
    lines.append("")
    return "\n".join(lines)


def join_rows(rows: list[dict]) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{row['label']} ({row['count']})" for row in rows)


if __name__ == "__main__":
    main()
