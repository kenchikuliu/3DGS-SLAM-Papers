#!/usr/bin/env python3
"""
Apply manual category overrides to the recategorized graph and regenerate
reviewed outputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized"
INPUT_JSON = BASE_DIR / "graphify_filtered_recategorized.json"
OUT_DIR = BASE_DIR / "final_reviewed"
OVERRIDES_CSV = OUT_DIR / "manual_category_overrides.csv"

OVERRIDES = [
    ("paper_2407_19323v6_4", "General", "MSP-MVS is multi-view stereo / reconstruction, not SLAM or robotics."),
    ("paper_2508_03077v1", "General", "RobustGS is feedforward 3DGS quality enhancement."),
    ("paper_2602_18322v1_1", "General", "Color/lightness correction for novel view synthesis."),
    ("paper_2602_18322v1", "General", "Color/lightness correction for novel view synthesis."),
    ("paper_2603_04869v1", "Robotics", "Feature matching paper framed around robotic vision problems."),
    ("paper_2603_26599v1_1", "General", "World-consistent video generation, not SLAM/robotics."),
    ("paper_2603_26599v1", "General", "World-consistent video generation, not SLAM/robotics."),
    ("paper_illumination_refinement_via_textual_cues_a_prompt_driven_approach_for_low_light_nerf_enhancement", "General", "Low-light NeRF enhancement."),
    ("paper_2506_06909", "SLAM", "GaME is an RGBD mapping method for evolving scenes."),
    ("paper_improving_orb_slam3_performance_in_textureless_environments_mapping_and_localization_in_hospital_corridors", "SLAM", "Direct ORB-SLAM3 mapping/localization paper."),
    ("paper_2403_20159", "SLAM", "HGS-Mapping is online dense mapping in urban scenes."),
    ("paper_2410_17084", "SLAM", "LiDAR-inertial-visual mapping with Gaussian splatting."),
    ("paper_2412_08496v2", "SLAM", "Drift-free Visual SLAM using digital twins."),
    ("paper_2502_19592", "Robotics", "Asynchronous multi-agent neural implicit mapping for robots."),
    ("paper_2503_12572", "SLAM", "Deblur Gaussian Splatting SLAM."),
    ("paper_2512_08625", "SLAM", "OpenMonoGS-SLAM is explicitly a SLAM framework."),
    ("paper_2512_22771", "Robotics", "Next-best-view for embodied agents / active learning."),
    ("paper_2603_20443", "SLAM", "TRGS-SLAM for thermal images."),
    ("paper_2310_06984", "SLAM", "Uncertainty-aware visual localization with NeRF."),
    ("paper_2401_14857", "SLAM", "LiDAR-inertial-visual fusion map rendering / mapping system."),
    ("paper_2406_11766", "SLAM", "Efficient and scalable localization with NeRF."),
    ("paper_2408_11966", "SLAM", "Visual localization in 3D maps is better grouped under SLAM/localization."),
    ("paper_2507_16144", "General", "LongSplat is online long-sequence 3DGS reconstruction."),
]


def load_graph() -> dict:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def slug(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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
            node_by_id[node_id]["manual_override_reason"] = reason

    category_nodes = {node["label"]: node["id"] for node in nodes if node.get("id", "").startswith("category_")}
    for label in ["General", "SLAM", "Robotics", "SLAM-Supplement", "Unknown"]:
        if label not in category_nodes:
            node_id = f"category_{slug(label)}"
            nodes.append({"id": node_id, "label": label, "file_type": "concept", "source_file": "manual_override"})
            category_nodes[label] = node_id

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
        "manual_override_count": len(OVERRIDES),
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
    }

    (OUT_DIR / "graphify_final_reviewed.json").write_text(
        json.dumps(reviewed_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_final_reviewed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "topic_clusters_final_reviewed.md").write_text(
        render_cluster_markdown(report),
        encoding="utf-8",
    )
    with open(OVERRIDES_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "reviewed_category", "reason"])
        writer.writeheader()
        for node_id, category, reason in OVERRIDES:
            writer.writerow({"id": node_id, "reviewed_category": category, "reason": reason})

    summary = {
        "manual_override_count": len(OVERRIDES),
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
