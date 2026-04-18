#!/usr/bin/env python3
"""
Export merged graph JSON into CSV formats that are easy to import into Neo4j,
Gephi, or spreadsheet tooling, plus a lightweight analysis summary.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
GRAPH_DIR = ROOT / "graphify-out"
MERGED_JSON = GRAPH_DIR / "graphify_merged.json"


def load_graph() -> dict:
    return json.loads(MERGED_JSON.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    graph = load_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])

    node_rows = []
    for node in nodes:
        node_rows.append(
            {
                "id": node.get("id", ""),
                "label": node.get("label", ""),
                "file_type": node.get("file_type", ""),
                "category": node.get("category", ""),
                "source_file": node.get("source_file", ""),
                "node_count": len(node.get("nodes", [])) if isinstance(node.get("nodes"), list) else "",
            }
        )

    edge_rows = []
    degrees = Counter()
    relation_counts = Counter()
    file_type_counts = Counter()
    category_counts = Counter()
    node_by_id = {node["id"]: node for node in nodes if "id" in node}
    doc_out_degree = Counter()

    for node in nodes:
        file_type_counts[node.get("file_type", "unknown")] += 1
        if node.get("category"):
            category_counts[node["category"]] += 1

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        relation = edge.get("relation", "")
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": edge.get("confidence", ""),
                "weight": edge.get("weight", ""),
                "source_file": edge.get("source_file", ""),
            }
        )
        degrees[source] += 1
        degrees[target] += 1
        relation_counts[relation] += 1
        if node_by_id.get(source, {}).get("file_type") == "document":
            doc_out_degree[source] += 1

    hyperedge_rows = []
    for hyperedge in hyperedges:
        hyperedge_rows.append(
            {
                "id": hyperedge.get("id", ""),
                "label": hyperedge.get("label", ""),
                "relation": hyperedge.get("relation", ""),
                "confidence": hyperedge.get("confidence", ""),
                "confidence_score": hyperedge.get("confidence_score", ""),
                "member_count": len(hyperedge.get("nodes", [])),
                "source_file": hyperedge.get("source_file", ""),
                "members": " | ".join(hyperedge.get("nodes", [])),
            }
        )

    top_nodes = []
    for node_id, degree in degrees.most_common(50):
        node = node_by_id.get(node_id, {})
        top_nodes.append(
            {
                "id": node_id,
                "label": node.get("label", ""),
                "file_type": node.get("file_type", ""),
                "category": node.get("category", ""),
                "degree": degree,
                "source_file": node.get("source_file", ""),
            }
        )

    top_documents = []
    for node_id, out_degree in doc_out_degree.most_common(50):
        node = node_by_id.get(node_id, {})
        top_documents.append(
            {
                "id": node_id,
                "label": node.get("label", ""),
                "category": node.get("category", ""),
                "out_degree": out_degree,
                "source_file": node.get("source_file", ""),
            }
        )

    relation_rows = [{"relation": k, "count": v} for k, v in relation_counts.most_common()]
    file_type_rows = [{"file_type": k, "count": v} for k, v in file_type_counts.most_common()]
    category_rows = [{"category": k, "count": v} for k, v in category_counts.most_common()]

    write_csv(
        GRAPH_DIR / "graph_nodes.csv",
        node_rows,
        ["id", "label", "file_type", "category", "source_file", "node_count"],
    )
    write_csv(
        GRAPH_DIR / "graph_edges.csv",
        edge_rows,
        ["source", "target", "relation", "confidence", "weight", "source_file"],
    )
    write_csv(
        GRAPH_DIR / "graph_hyperedges.csv",
        hyperedge_rows,
        ["id", "label", "relation", "confidence", "confidence_score", "member_count", "source_file", "members"],
    )
    write_csv(
        GRAPH_DIR / "graph_top_nodes.csv",
        top_nodes,
        ["id", "label", "file_type", "category", "degree", "source_file"],
    )
    write_csv(
        GRAPH_DIR / "graph_top_documents.csv",
        top_documents,
        ["id", "label", "category", "out_degree", "source_file"],
    )
    write_csv(GRAPH_DIR / "graph_relation_counts.csv", relation_rows, ["relation", "count"])
    write_csv(GRAPH_DIR / "graph_file_type_counts.csv", file_type_rows, ["file_type", "count"])
    write_csv(GRAPH_DIR / "graph_category_counts.csv", category_rows, ["category", "count"])

    summary = {
        "source_markdown_count": graph.get("source_markdown_count", 0),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "hyperedge_count": len(hyperedges),
        "top_relations": relation_rows[:10],
        "top_file_types": file_type_rows[:10],
        "top_categories": category_rows[:10],
        "top_nodes": top_nodes[:15],
        "top_documents": top_documents[:15],
    }
    (GRAPH_DIR / "graph_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
