#!/usr/bin/env python3
"""
Build a deterministic paper graph from markdown files and merge any existing
chunk-based Graphify JSON outputs into one consolidated graph.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
MD_DIR = ROOT / "extracted_markdown"
OUT_DIR = ROOT / "graphify-out"

RESULT_GLOB = ".graphify_chunk_*_result.json"

DATASETS = [
    "TUM-RGBD",
    "Replica",
    "EuRoC",
    "EuRoC MAV",
    "ScanNet",
    "ScanNet++",
    "Matterport3D",
    "MP3D",
    "HM3D",
    "Gibson",
    "7-Scenes",
    "KITTI",
    "nuScenes",
    "CO3D",
    "DTU",
]

METRICS = [
    "ATE",
    "ATE RMSE",
    "RPE",
    "PSNR",
    "SSIM",
    "LPIPS",
    "SPL",
    "Chamfer",
    "IoU",
    "Accuracy",
    "Recall",
]

METHODS = [
    ("3D Gaussian Splatting", ["3d gaussian splatting", "gaussian splatting", "3dgs"]),
    ("NeRF", ["nerf", "neural radiance field", "neural radiance fields"]),
    ("SLAM", ["slam"]),
    ("Visual Localization", ["visual localization", "camera relocalization", "relocalization"]),
    ("Navigation", ["navigation", "path planning"]),
    ("Reconstruction", ["reconstruction", "scene reconstruction"]),
    ("Semantic Mapping", ["semantic mapping", "open-vocabulary"]),
    ("Pose Graph Optimization", ["pose graph optimization", "pgo"]),
    ("IMU Integration", ["imu", "inertial"]),
    ("LiDAR", ["lidar"]),
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def md_files() -> list[Path]:
    return sorted(p for p in MD_DIR.glob("*.md") if p.name != "save_chunks.py")


def category_from_name(name: str) -> str:
    match = re.search(r"\[([^\]]+)\]", name)
    return match.group(1).strip() if match else "Unknown"


def label_from_name(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"^\d+_", "", base)
    base = re.sub(r"^\[[^\]]+\]_", "", base)
    return base.strip()


def add_node(nodes: dict[str, dict], node: dict) -> None:
    existing = nodes.get(node["id"])
    if existing is None:
        nodes[node["id"]] = node
        return
    for key, value in node.items():
        if existing.get(key) in (None, "", []):
            existing[key] = value


def add_edge(edges: dict[tuple[str, str, str], dict], edge: dict) -> None:
    key = (edge["source"], edge["target"], edge["relation"])
    existing = edges.get(key)
    if existing is None or edge.get("weight", 0) > existing.get("weight", 0):
        edges[key] = edge


def build_deterministic_graph() -> tuple[dict[str, dict], dict[tuple[str, str, str], dict]]:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    doc_ids: list[str] = []
    category_groups: dict[str, list[str]] = {}
    concept_counts: Counter[str] = Counter()

    for md in md_files():
        text = md.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        doc_id = f"paper_{slug(md.stem)}"
        doc_ids.append(doc_id)
        category = category_from_name(md.name)
        label = label_from_name(md.name)

        add_node(
            nodes,
            {
                "id": doc_id,
                "label": label,
                "file_type": "document",
                "source_file": md.name,
                "category": category,
            },
        )

        cat_id = f"category_{slug(category)}"
        add_node(
            nodes,
            {
                "id": cat_id,
                "label": category,
                "file_type": "concept",
                "source_file": md.name,
            },
        )
        add_edge(
            edges,
            {
                "source": doc_id,
                "target": cat_id,
                "relation": "belongs_to",
                "confidence": "EXTRACTED",
                "weight": 1.0,
                "source_file": md.name,
            },
        )
        category_groups.setdefault(category, []).append(doc_id)

        for method_label, patterns in METHODS:
            if any(p in lowered for p in patterns):
                method_id = f"concept_{slug(method_label)}"
                add_node(
                    nodes,
                    {
                        "id": method_id,
                        "label": method_label,
                        "file_type": "concept",
                        "source_file": md.name,
                    },
                )
                add_edge(
                    edges,
                    {
                        "source": doc_id,
                        "target": method_id,
                        "relation": "references",
                        "confidence": "EXTRACTED",
                        "weight": 1.0,
                        "source_file": md.name,
                    },
                )
                concept_counts[method_id] += 1

        for dataset in DATASETS:
            if dataset.lower() in lowered:
                ds_id = f"dataset_{slug(dataset)}"
                add_node(
                    nodes,
                    {
                        "id": ds_id,
                        "label": dataset,
                        "file_type": "concept",
                        "source_file": md.name,
                    },
                )
                add_edge(
                    edges,
                    {
                        "source": doc_id,
                        "target": ds_id,
                        "relation": "shares_data_with",
                        "confidence": "EXTRACTED",
                        "weight": 1.0,
                        "source_file": md.name,
                    },
                )

        for metric in METRICS:
            if metric.lower() in lowered:
                metric_id = f"metric_{slug(metric)}"
                add_node(
                    nodes,
                    {
                        "id": metric_id,
                        "label": metric,
                        "file_type": "concept",
                        "source_file": md.name,
                    },
                )
                add_edge(
                    edges,
                    {
                        "source": doc_id,
                        "target": metric_id,
                        "relation": "evaluates_with",
                        "confidence": "EXTRACTED",
                        "weight": 1.0,
                        "source_file": md.name,
                    },
                )

    for category, members in category_groups.items():
        if len(members) < 2:
            continue
        group_id = f"hyperedge_{slug(category)}"
        add_node(
            nodes,
            {
                "id": group_id,
                "label": f"{category} paper cluster",
                "file_type": "hyperedge",
                "source_file": "deterministic",
                "nodes": members,
            },
        )

    return nodes, edges


def merge_existing_results(nodes: dict[str, dict], edges: dict[tuple[str, str, str], dict]) -> list[dict]:
    chunk_summaries = []
    for path in sorted(MD_DIR.glob(RESULT_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunk_summaries.append(
            {
                "file": path.name,
                "nodes": len(data.get("nodes", [])),
                "edges": len(data.get("edges", [])),
                "hyperedges": len(data.get("hyperedges", [])),
            }
        )
        for node in data.get("nodes", []):
            add_node(nodes, node)
        for hyperedge in data.get("hyperedges", []):
            hyperedge_node = {
                "id": hyperedge["id"],
                "label": hyperedge["label"],
                "file_type": "hyperedge",
                "source_file": hyperedge.get("source_file"),
                "nodes": hyperedge.get("nodes", []),
            }
            add_node(nodes, hyperedge_node)
        for edge in data.get("edges", []):
            add_edge(edges, edge)
    return chunk_summaries


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    nodes, edges = build_deterministic_graph()
    chunk_summaries = merge_existing_results(nodes, edges)

    hyperedges = []
    for node in nodes.values():
        if node.get("file_type") == "hyperedge":
            hyperedges.append(
                {
                    "id": node["id"],
                    "label": node["label"],
                    "nodes": node.get("nodes", []),
                    "relation": "participate_in",
                    "confidence": "INFERRED",
                    "confidence_score": 0.7,
                    "source_file": node.get("source_file"),
                }
            )

    payload = {
        "generated_at": time_stamp(),
        "source_markdown_count": len(md_files()),
        "nodes": sorted(nodes.values(), key=lambda x: x["id"]),
        "edges": sorted(edges.values(), key=lambda x: (x["source"], x["target"], x["relation"])),
        "hyperedges": sorted(hyperedges, key=lambda x: x["id"]),
        "merged_chunk_results": chunk_summaries,
    }
    (OUT_DIR / "graphify_merged.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_markdown_count": payload["source_markdown_count"],
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "hyperedge_count": len(payload["hyperedges"]),
        "merged_chunk_result_count": len(chunk_summaries),
    }
    (OUT_DIR / "graphify_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def time_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
