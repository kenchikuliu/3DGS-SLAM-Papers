#!/usr/bin/env python3
"""
Apply a third batch of manual category overrides on top of final_reviewed_v2 and
generate final_reviewed_v3 outputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2"
INPUT_JSON = BASE_DIR / "graphify_final_reviewed_v2.json"
OUT_DIR = BASE_DIR / "final_reviewed_v3"
OVERRIDES_CSV = OUT_DIR / "manual_category_overrides_v3.csv"

OVERRIDES = [
    ("paper_2503_03373", "SLAM", "Direct sparse odometry with continuous Gaussian maps is odometry/localization."),
    ("paper_2403_12535", "SLAM", "Explicitly an RGBD SLAM system."),
    ("paper_2508_01150", "Robotics", "Open-vocabulary dense mapping for robotic/VR-AR scene interaction."),
    ("paper_2503_16710", "SLAM", "Explicitly 4D Gaussian Splatting SLAM."),
    ("paper_2512_01296", "General", "Efficient reconstruction with Gaussian surfels is reconstruction-centric."),
    ("paper_2509_12702", "Robotics", "Multi-robot neural implicit mapping under communication constraints."),
    ("paper_2510_12749", "SLAM", "Panoptic odometry / tracking / rendering for urban scenes fits SLAM."),
    ("paper_2604_03092v1", "SLAM", "Monocular SLAM with feed-forward Gaussian splatting."),
    ("paper_2406_12202", "SLAM", "Fast global localization on NeRF maps is localization-focused."),
    ("paper_2408_04268", "General", "Evaluation paper comparing reconstruction approaches."),
    ("paper_2511_13571", "General", "3DGS optimization framework, not SLAM/robotics-specific."),
    ("paper_2510_23988", "Robotics", "Survey on collaborative multi-robot SLAM with 3DGS."),
    ("paper_2510_06644", "SLAM", "Explicitly real-time 3DGS SLAM."),
    ("paper_2509_00741", "SLAM", "DyPho-SLAM is clearly SLAM."),
    ("paper_2309_13240", "Robotics", "Field-of-view extrapolation motivated by robotic navigation."),
    ("paper_2310_00684", "Robotics", "View planning for unknown-object reconstruction with autonomous robots."),
    ("paper_2412_20056", "SLAM", "GSplatLoc is camera localization."),
    ("paper_2503_23480", "Robotics", "Indoor localization for mobile robots in known maps."),
    ("paper_2503_01646", "SLAM", "OpenGS-SLAM is explicitly dense semantic SLAM."),
    ("paper_2602_17182", "SLAM", "NRGS-SLAM is explicitly non-rigid SLAM."),
    ("paper_tvg_slam_robust_gaussian_splatting_slam_with_tri_view_geometric_constraints_1", "SLAM", "TVG-SLAM is explicitly SLAM."),
    ("paper_2507_23677", "SLAM", "Stereo 3DGS SLAM for outdoor urban scenes."),
    ("paper_2506_21420", "SLAM", "EndoFlow-SLAM is explicitly endoscopic SLAM."),
    ("paper_2509_00741v1_4", "SLAM", "Duplicate DyPho-SLAM entry."),
    ("paper_2502_05752", "Robotics", "Implicit neural map representation for robotics-oriented mapping/localization."),
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
            node_by_id[node_id]["manual_override_v3_reason"] = reason

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
        "manual_override_count_v3": len(OVERRIDES),
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
    }

    (OUT_DIR / "graphify_final_reviewed_v3.json").write_text(
        json.dumps(reviewed_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_final_reviewed_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_final_reviewed_v3.md").write_text(
        render_cluster_markdown(report),
        encoding="utf-8",
    )
    with open(OVERRIDES_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "reviewed_category", "reason"])
        writer.writeheader()
        for node_id, category, reason in OVERRIDES:
            writer.writerow({"id": node_id, "reviewed_category": category, "reason": reason})

    summary = {
        "manual_override_count_v3": len(OVERRIDES),
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
