#!/usr/bin/env python3
"""
Infer categories for documents currently labeled Unknown in the filtered graph,
then regenerate a recategorized graph and topic-cluster report.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
MD_DIR = ROOT / "extracted_markdown"
FILTERED_DIR = ROOT / "graphify-out" / "filtered"
FILTERED_JSON = FILTERED_DIR / "graphify_filtered.json"
OUT_DIR = FILTERED_DIR / "recategorized"

SLAM_PATTERNS = [
    " slam",
    "simultaneous localization and mapping",
    "visual localization",
    "camera relocalization",
    "pose graph",
    "odometry",
    "loop closure",
    "rgb-d",
]

ROBOTICS_PATTERNS = [
    "robot",
    "robotic",
    "navigation",
    "grasp",
    "manipulation",
    "exploration",
    "embodied",
    "drone",
    "locomotion",
]

SUPPLEMENT_PATTERNS = [
    "supplementary",
    "supplement",
    "appendix",
    "technical report",
]


def load_graph() -> dict:
    return json.loads(FILTERED_JSON.read_text(encoding="utf-8"))


def read_markdown(stem: str) -> str:
    path = MD_DIR / f"{stem}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def infer_category(text: str, source_file: str) -> tuple[str, str]:
    lower = f"{source_file}\n{text[:15000]}".lower()
    slam_score = score_patterns(lower, SLAM_PATTERNS)
    robotics_score = score_patterns(lower, ROBOTICS_PATTERNS)
    supplement_score = score_patterns(lower, SUPPLEMENT_PATTERNS)

    if supplement_score >= 2:
        return "SLAM-Supplement", f"supplement_score={supplement_score}"
    if slam_score >= robotics_score + 1 and slam_score >= 2:
        return "SLAM", f"slam_score={slam_score}, robotics_score={robotics_score}"
    if robotics_score >= slam_score + 1 and robotics_score >= 2:
        return "Robotics", f"robotics_score={robotics_score}, slam_score={slam_score}"
    if slam_score >= 1 and robotics_score >= 1:
        return "Robotics", f"mixed_scores robotics={robotics_score}, slam={slam_score}"
    return "General", f"default_general slam={slam_score}, robotics={robotics_score}"


def score_patterns(text: str, patterns: list[str]) -> int:
    return sum(text.count(pattern) for pattern in patterns)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])
    node_by_id = {node["id"]: node for node in nodes}

    unknown_category_targets = {
        node["id"] for node in nodes if node.get("label") == "Unknown" and node.get("file_type") == "concept"
    }
    docs_pointing_to_unknown = {
        edge.get("source")
        for edge in edges
        if edge.get("relation") == "belongs_to" and edge.get("target") in unknown_category_targets
    }

    inferred_rows = []
    unknown_doc_ids = sorted(
        {
            node["id"]
            for node in nodes
            if node.get("category") == "Unknown" or node.get("id") in docs_pointing_to_unknown
        }
    )

    for doc_id in unknown_doc_ids:
        node = node_by_id[doc_id]
        stem = Path(node.get("source_file", "")).stem
        text = read_markdown(stem)
        inferred_category, rationale = infer_category(text, node.get("source_file", ""))
        node["inferred_category"] = inferred_category
        inferred_rows.append(
            {
                "id": doc_id,
                "label": node.get("label", ""),
                "source_file": node.get("source_file", ""),
                "old_category": node.get("category", ""),
                "new_category": inferred_category,
                "rationale": rationale,
            }
        )

    unknown_doc_id_set = set(unknown_doc_ids)

    for node in nodes:
        if node.get("file_type") == "document" and node.get("id") in unknown_doc_id_set:
            node["category"] = node.get("inferred_category", "General")
        elif node.get("id") in unknown_doc_id_set:
            node["category"] = node.get("inferred_category", "General")

    old_category_nodes = unknown_category_targets
    edges = [
        edge
        for edge in edges
        if not (
            edge.get("relation") == "belongs_to"
            and (
                edge.get("target") in old_category_nodes
                or edge.get("source") in unknown_doc_id_set
            )
        )
    ]

    category_node_map = {}
    for name in ["General", "SLAM", "Robotics", "SLAM-Supplement", "Unknown"]:
        node_id = f"category_{slug(name)}"
        category_node_map[name] = node_id
        if not any(node.get("id") == node_id for node in nodes):
            nodes.append(
                {
                    "id": node_id,
                    "label": name,
                    "file_type": "concept",
                    "source_file": "inferred",
                }
            )

    existing_belongs = {
        (edge.get("source"), edge.get("target"))
        for edge in edges
        if edge.get("relation") == "belongs_to"
    }
    for node in nodes:
        if node.get("file_type") not in {"document", "paper"}:
            continue
        target = category_node_map.get(node.get("category", "Unknown"), category_node_map["Unknown"])
        key = (node["id"], target)
        if key not in existing_belongs:
            edges.append(
                {
                    "source": node["id"],
                    "target": target,
                    "relation": "belongs_to",
                    "confidence": "INFERRED",
                    "weight": 1.0,
                    "source_file": node.get("source_file", ""),
                }
            )

    node_by_id = {node["id"]: node for node in nodes}
    edges = [
        edge
        for edge in edges
        if not (
            edge.get("relation") == "belongs_to"
            and node_by_id.get(edge.get("target"), {}).get("label") == "Unknown"
        )
    ]

    report = build_cluster_report(nodes, edges, hyperedges)
    recategorized_graph = {
        "generated_from": str(FILTERED_JSON),
        "source_markdown_count": graph.get("source_markdown_count", 0),
        "unknown_document_count_before": len(unknown_doc_ids),
        "unknown_document_count_after": sum(
            1 for node in nodes if node.get("file_type") == "document" and node.get("category") == "Unknown"
        ),
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
    }

    (OUT_DIR / "graphify_filtered_recategorized.json").write_text(
        json.dumps(recategorized_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_recategorized.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_recategorized.md").write_text(
        render_cluster_markdown(report),
        encoding="utf-8",
    )

    with open(OUT_DIR / "unknown_category_inference.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "label", "source_file", "old_category", "new_category", "rationale"],
        )
        writer.writeheader()
        writer.writerows(inferred_rows)

    summary = {
        "unknown_document_count_before": len(unknown_doc_ids),
        "unknown_document_count_after": recategorized_graph["unknown_document_count_after"],
        "new_category_counts_for_unknown_docs": dict(Counter(row["new_category"] for row in inferred_rows)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def build_cluster_report(nodes: list[dict], edges: list[dict], hyperedges: list[dict]) -> dict:
    node_by_id = {node["id"]: node for node in nodes}
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
        if relation == "belongs_to" and source in node_by_id and node_by_id[source].get("file_type") == "document":
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
                if relation == "belongs_to":
                    continue
                target_label = target_node.get("label", "")
                target_id = target_node.get("id", "")
                target_type = target_node.get("file_type", "")
                if target_id.startswith("dataset_") or target_type == "dataset":
                    dataset_counts[target_label] += 1
                elif target_id.startswith("metric_") or target_type == "metric":
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
    lines = ["# Topic Clusters", "", "## Category Sizes", ""]
    for row in report.get("category_sizes", []):
        lines.append(f"- {row['category']}: {row['document_count']}")

    lines.extend(["", "## Relation Counts", ""])
    for row in report.get("relation_counts", []):
        lines.append(f"- {row['label']}: {row['count']}")

    for category in report.get("categories", []):
        lines.extend(
            [
                "",
                f"## {category['category']}",
                "",
                f"- documents: {category['document_count']}",
                "- top concepts: " + join_rows(category["top_concepts"]),
                "- top datasets: " + join_rows(category["top_datasets"]),
                "- top metrics: " + join_rows(category["top_metrics"]),
                "- representative documents:",
            ]
        )
        for doc in category["representative_documents"]:
            lines.append(f"  - {doc['label']} ({doc['degree']})")

    lines.extend(["", "## Hyperedges", "", f"- count: {report.get('hyperedge_count', 0)}", ""])
    return "\n".join(lines)


def join_rows(rows: list[dict]) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{row['label']} ({row['count']})" for row in rows)


if __name__ == "__main__":
    main()
