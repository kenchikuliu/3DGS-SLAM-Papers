#!/usr/bin/env python3
"""
Deduplicate the final_reviewed_v3 graph by merging obvious duplicate document
entries that differ only by version suffixes or duplicate filename markers.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2" / "final_reviewed_v3"
INPUT_JSON = BASE_DIR / "graphify_final_reviewed_v3.json"
OUT_DIR = BASE_DIR / "deduped"


def load_graph() -> dict:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def canonical_source_key(source_file: str) -> str:
    stem = Path(source_file).stem.lower()
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"v\d+$", "", stem)
    stem = re.sub(r"v\d+\s*$", "", stem)
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_")
    return stem


def merge_node(primary: dict, duplicate: dict) -> None:
    alias_ids = set(primary.get("alias_ids", []))
    alias_ids.add(duplicate["id"])
    if duplicate.get("alias_ids"):
        alias_ids.update(duplicate["alias_ids"])
    primary["alias_ids"] = sorted(alias_ids)

    alias_files = set(primary.get("alias_source_files", []))
    alias_files.add(duplicate.get("source_file", ""))
    if duplicate.get("alias_source_files"):
        alias_files.update(duplicate["alias_source_files"])
    primary["alias_source_files"] = sorted(x for x in alias_files if x)

    if not primary.get("source_file") and duplicate.get("source_file"):
        primary["source_file"] = duplicate["source_file"]
    if not primary.get("label") and duplicate.get("label"):
        primary["label"] = duplicate["label"]


def choose_primary(nodes: list[dict]) -> dict:
    def score(node: dict) -> tuple:
        src = node.get("source_file", "")
        return (
            0 if re.search(r"\(\d+\)\.md$", src) else 1,
            0 if re.search(r"v\d+\.md$", src.lower()) else 1,
            -len(src),
            node.get("id", ""),
        )
    return sorted(nodes, key=score, reverse=True)[0]


def build_report(nodes: list[dict], edges: list[dict], hyperedges: list[dict]) -> dict:
    node_by_id = {node["id"]: node for node in nodes}
    relation_counts = Counter(edge.get("relation", "") for edge in edges)
    category_docs = defaultdict(list)

    for edge in edges:
        if edge.get("relation") == "belongs_to":
            src = edge.get("source")
            tgt = edge.get("target")
            if src in node_by_id and tgt in node_by_id and node_by_id[src].get("file_type") in {"document", "paper"}:
                category_docs[node_by_id[tgt].get("label", "Unknown")].append(src)

    return {
        "category_sizes": sorted(
            [{"category": k, "document_count": len(v)} for k, v in category_docs.items() if k != "Unknown"],
            key=lambda x: (-x["document_count"], x["category"]),
        ),
        "relation_counts": [{"label": k, "count": v} for k, v in relation_counts.most_common()],
        "hyperedge_count": len(hyperedges),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])

    doc_nodes = [n for n in nodes if n.get("file_type") in {"document", "paper"}]
    other_nodes = [n for n in nodes if n.get("file_type") not in {"document", "paper"}]

    groups = defaultdict(list)
    for node in doc_nodes:
        groups[canonical_source_key(node.get("source_file", node.get("label", node["id"])))].append(node)

    alias_rows = []
    id_map = {}
    deduped_doc_nodes = []

    for key, group in groups.items():
        primary = dict(choose_primary(group))
        primary.setdefault("alias_ids", [])
        primary.setdefault("alias_source_files", [])
        id_map[primary["id"]] = primary["id"]
        for duplicate in group:
            if duplicate["id"] == primary["id"]:
                continue
            merge_node(primary, duplicate)
            id_map[duplicate["id"]] = primary["id"]
            alias_rows.append(
                {
                    "canonical_id": primary["id"],
                    "canonical_label": primary.get("label", ""),
                    "duplicate_id": duplicate["id"],
                    "duplicate_label": duplicate.get("label", ""),
                    "canonical_source_file": primary.get("source_file", ""),
                    "duplicate_source_file": duplicate.get("source_file", ""),
                    "canonical_key": key,
                }
            )
        deduped_doc_nodes.append(primary)

    deduped_nodes = deduped_doc_nodes + other_nodes
    seen_edges = set()
    deduped_edges = []
    for edge in edges:
        source = id_map.get(edge.get("source"), edge.get("source"))
        target = id_map.get(edge.get("target"), edge.get("target"))
        item = dict(edge)
        item["source"] = source
        item["target"] = target
        key = (source, target, item.get("relation"), item.get("source_file", ""))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_edges.append(item)

    deduped_hyperedges = []
    for hyperedge in hyperedges:
        item = dict(hyperedge)
        item["nodes"] = sorted({id_map.get(node_id, node_id) for node_id in hyperedge.get("nodes", [])})
        deduped_hyperedges.append(item)

    deduped_graph = {
        "generated_from": str(INPUT_JSON),
        "document_nodes_before": len(doc_nodes),
        "document_nodes_after": len(deduped_doc_nodes),
        "alias_merge_count": len(alias_rows),
        "nodes": deduped_nodes,
        "edges": deduped_edges,
        "hyperedges": deduped_hyperedges,
    }
    (OUT_DIR / "graphify_final_reviewed_v3_deduped.json").write_text(
        json.dumps(deduped_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(OUT_DIR / "alias_map.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "canonical_id",
                "canonical_label",
                "duplicate_id",
                "duplicate_label",
                "canonical_source_file",
                "duplicate_source_file",
                "canonical_key",
            ],
        )
        writer.writeheader()
        writer.writerows(alias_rows)

    report = build_report(deduped_nodes, deduped_edges, deduped_hyperedges)
    summary = {
        "document_nodes_before": len(doc_nodes),
        "document_nodes_after": len(deduped_doc_nodes),
        "alias_merge_count": len(alias_rows),
        "node_count_after": len(deduped_nodes),
        "edge_count_after": len(deduped_edges),
        "category_sizes": report["category_sizes"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
