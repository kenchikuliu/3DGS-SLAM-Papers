#!/usr/bin/env python3
"""
Apply a second batch of manual category overrides on top of final_reviewed and
generate final_reviewed_v2 outputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed"
INPUT_JSON = BASE_DIR / "graphify_final_reviewed.json"
OUT_DIR = BASE_DIR / "final_reviewed_v2"
OVERRIDES_CSV = OUT_DIR / "manual_category_overrides_v2.csv"

OVERRIDES = [
    ("paper_2405_14824", "SLAM", "Camera relocalization with NeRF belongs with localization/SLAM."),
    ("paper_2408_12677", "SLAM", "Online RGB-D mapping with Gaussian/TSDF fusion."),
    ("paper_2409_12899", "General", "LiDAR-incorporated large-scale reconstruction is reconstruction-centric."),
    ("paper_2509_01228", "Robotics", "Multi-agent distributed mapping for robots."),
    ("paper_3746027_3755375", "General", "Low-light scene reconstruction and enhancement."),
    ("paper_2406_06948", "Robotics", "Uncertainty-driven active mapping fits robotics exploration."),
    ("paper_2504_10331v3", "General", "Low-light scene reconstruction and enhancement."),
    ("paper_2508_17876", "SLAM", "Camera pose refinement via 3DGS is localization-focused."),
    ("paper_2601_00705", "SLAM", "RGS-SLAM is explicitly SLAM."),
    ("paper_2408_03825", "SLAM", "Photometric SLAM acceleration paper."),
    ("paper_2502_13803", "SLAM", "3DGS-aided localization using a visual SLAM map."),
    ("paper_2503_09447", "General", "Online language splatting is semantic scene representation, not clearly SLAM/robotics."),
    ("paper_2509_00433", "SLAM", "AGS is explicitly Gaussian Splatting SLAM."),
    ("paper_2310_00685", "Robotics", "Active one-shot view planning for autonomous robots."),
    ("paper_2503_05425", "SLAM", "LiDAR-enhanced 3DGS mapping with pose tracking."),
    ("paper_2510_09962", "Robotics", "Variation-aware online mapping in semi-static scenes leans robotics deployment."),
    ("paper_2602_21644", "SLAM", "DAGS-SLAM is explicitly dynamic-aware SLAM."),
]


def load_graph() -> dict:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])
    node_by_id = {node["id"]: node for node in nodes}
    override_map = {row[0]: row for row in OVERRIDES}

    for node_id, category, reason in OVERRIDES:
        if node_id in node_by_id:
            node_by_id[node_id]["category"] = category
            node_by_id[node_id]["manual_override_v2_reason"] = reason

    category_nodes = {node["label"]: node["id"] for node in nodes if node.get("id", "").startswith("category_")}
    edges = [
        edge
        for edge in edges
        if not (edge.get("relation") == "belongs_to" and edge.get("source") in override_map)
    ]
    for node_id, category, _ in OVERRIDES:
        if node_id in node_by_id:
            edges.append(
                {
                    "source": node_id,
                    "target": category_nodes[category],
                    "relation": "belongs_to",
                    "confidence": "MANUAL",
                    "weight": 1.0,
                    "source_file": node_by_id[node_id].get("source_file", ""),
                }
            )

    report = build_cluster_report(nodes, edges, hyperedges)
    reviewed_graph = {
        "generated_from": str(INPUT_JSON),
        "manual_override_count_v2": len(OVERRIDES),
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
    }

    (OUT_DIR / "graphify_final_reviewed_v2.json").write_text(
        json.dumps(reviewed_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_final_reviewed_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_final_reviewed_v2.md").write_text(
        render_cluster_markdown(report),
        encoding="utf-8",
    )
    with open(OVERRIDES_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "reviewed_category", "reason"])
        writer.writeheader()
        for node_id, category, reason in OVERRIDES:
            writer.writerow({"id": node_id, "reviewed_category": category, "reason": reason})

    summary = {
        "manual_override_count_v2": len(OVERRIDES),
        "category_sizes": report["category_sizes"],
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
            if node_by_id[source].get("file_type") in {"document", "paper"}:
                doc_degree[source] += 1
        if relation == "belongs_to" and source in node_by_id and node_by_id[source].get("file_type") in {"document", "paper"}:
            category = node_by_id[target].get("label", "Unknown")
            category_docs[category].append(source)

    category_sizes = sorted(
        [{"category": category, "document_count": len(doc_ids)} for category, doc_ids in category_docs.items() if category != "Unknown"],
        key=lambda x: (-x["document_count"], x["category"]),
    )

    categories = []
    for category, doc_ids in sorted(category_docs.items(), key=lambda item: (-len(item[1]), item[0])):
        if category == "Unknown":
            continue
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
