#!/usr/bin/env python3
"""
Export a graph JSON into Neo4j-friendly CSV files plus APOC and native Cypher import files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
GRAPH_DIR = ROOT / "graphify-out"

def load_graph(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(GRAPH_DIR / "graphify_merged.json"))
    parser.add_argument("--output", default=str(GRAPH_DIR / "neo4j"))
    args = parser.parse_args()

    input_path = Path(args.input)
    neo4j_dir = Path(args.output)

    graph = load_graph(input_path)
    neo4j_dir.mkdir(parents=True, exist_ok=True)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    hyperedges = graph.get("hyperedges", [])

    node_rows = []
    hyperedge_ids = {h.get("id", "") for h in hyperedges}

    for node in nodes:
        node_id = node.get("id", "")
        file_type = node.get("file_type", "") or "Unknown"
        labels = ["Entity", sanitize_label(file_type)]
        if node_id in hyperedge_ids:
            labels.append("Hyperedge")
        if node.get("category"):
            labels.append(sanitize_label(node["category"]))

        node_rows.append(
            {
                "id:ID": node_id,
                "label": node.get("label", ""),
                "file_type": file_type,
                "category": node.get("category", ""),
                "source_file": node.get("source_file", ""),
                "labels:LABEL": ";".join(dict.fromkeys(labels)),
            }
        )

    edge_rows = []
    edge_types = set()
    for edge in edges:
        edge_type = sanitize_rel(edge.get("relation", "RELATED_TO"))
        edge_types.add(edge_type)
        edge_rows.append(
            {
                ":START_ID": edge.get("source", ""),
                ":END_ID": edge.get("target", ""),
                ":TYPE": edge_type,
                "relation": edge.get("relation", ""),
                "confidence": edge.get("confidence", ""),
                "weight:float": edge.get("weight", ""),
                "source_file": edge.get("source_file", ""),
            }
        )

    hyperedge_member_rows = []
    for hyperedge in hyperedges:
        hyper_id = hyperedge.get("id", "")
        for member in hyperedge.get("nodes", []):
            hyperedge_member_rows.append(
                {
                    ":START_ID": hyper_id,
                    ":END_ID": member,
                    ":TYPE": "HAS_MEMBER",
                    "relation": hyperedge.get("relation", "participate_in"),
                    "confidence": hyperedge.get("confidence", ""),
                    "weight:float": hyperedge.get("confidence_score", ""),
                    "source_file": hyperedge.get("source_file", ""),
                }
            )

    write_csv(
        neo4j_dir / "nodes.csv",
        node_rows,
        ["id:ID", "label", "file_type", "category", "source_file", "labels:LABEL"],
    )
    write_csv(
        neo4j_dir / "edges.csv",
        edge_rows,
        [":START_ID", ":END_ID", ":TYPE", "relation", "confidence", "weight:float", "source_file"],
    )
    write_csv(
        neo4j_dir / "hyperedge_members.csv",
        hyperedge_member_rows,
        [":START_ID", ":END_ID", ":TYPE", "relation", "confidence", "weight:float", "source_file"],
    )

    cypher = """// Copy nodes.csv, edges.csv, hyperedge_members.csv into Neo4j import directory first.
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CALL {
  WITH row
  CREATE (n:Entity {id: row['id:ID']})
  SET n.label = row.label,
      n.file_type = row.file_type,
      n.category = row.category,
      n.source_file = row.source_file
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Document' THEN [1] ELSE [] END | SET n:Document)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Concept' THEN [1] ELSE [] END | SET n:Concept)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Hyperedge' THEN [1] ELSE [] END | SET n:Hyperedge)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'General' THEN [1] ELSE [] END | SET n:General)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'SLAM' THEN [1] ELSE [] END | SET n:SLAM)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Robotics' THEN [1] ELSE [] END | SET n:Robotics)
  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'SLAM_Supplement' THEN [1] ELSE [] END | SET n:SLAM_Supplement)
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
CALL {
  WITH row
  MATCH (a:Entity {id: row[':START_ID']})
  MATCH (b:Entity {id: row[':END_ID']})
  CALL apoc.create.relationship(
    a,
    row[':TYPE'],
    {
      relation: row.relation,
      confidence: row.confidence,
      weight: toFloat(row['weight:float']),
      source_file: row.source_file
    },
    b
  ) YIELD rel
  RETURN rel
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///hyperedge_members.csv' AS row
CALL {
  WITH row
  MATCH (a:Entity {id: row[':START_ID']})
  MATCH (b:Entity {id: row[':END_ID']})
  MERGE (a)-[r:HAS_MEMBER]->(b)
  SET r.relation = row.relation,
      r.confidence = row.confidence,
      r.weight = toFloat(row['weight:float']),
      r.source_file = row.source_file
} IN TRANSACTIONS OF 1000 ROWS;
"""
    (neo4j_dir / "import.cypher").write_text(cypher, encoding="utf-8")
    (neo4j_dir / "import_native.cypher").write_text(
        build_native_cypher(sorted(edge_types)),
        encoding="utf-8",
    )

    summary = {
        "input_graph": str(input_path),
        "node_rows": len(node_rows),
        "edge_rows": len(edge_rows),
        "hyperedge_member_rows": len(hyperedge_member_rows),
        "neo4j_dir": str(neo4j_dir),
        "edge_types": sorted(edge_types),
        "apoc_import": str(neo4j_dir / "import.cypher"),
        "native_import": str(neo4j_dir / "import_native.cypher"),
    }
    (neo4j_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def sanitize_label(value: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    clean = clean.strip("_")
    return clean[:1].upper() + clean[1:] if clean else "Unknown"


def sanitize_rel(value: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in value.upper())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "RELATED_TO"


def build_native_cypher(edge_types: list[str]) -> str:
    lines = [
        "// Copy nodes.csv, edges.csv, hyperedge_members.csv into Neo4j import directory first.",
        "// Native import: no APOC dependency. Relationship types are expanded explicitly.",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS",
        "FOR (n:Entity) REQUIRE n.id IS UNIQUE;",
        "",
        "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row",
        "CALL {",
        "  WITH row",
        "  CREATE (n:Entity {id: row['id:ID']})",
        "  SET n.label = row.label,",
        "      n.file_type = row.file_type,",
        "      n.category = row.category,",
        "      n.source_file = row.source_file",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Document' THEN [1] ELSE [] END | SET n:Document)",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Concept' THEN [1] ELSE [] END | SET n:Concept)",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Hyperedge' THEN [1] ELSE [] END | SET n:Hyperedge)",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'General' THEN [1] ELSE [] END | SET n:General)",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'SLAM' THEN [1] ELSE [] END | SET n:SLAM)",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'Robotics' THEN [1] ELSE [] END | SET n:Robotics)",
        "  FOREACH (_ IN CASE WHEN row['labels:LABEL'] CONTAINS 'SLAM_Supplement' THEN [1] ELSE [] END | SET n:SLAM_Supplement)",
        "} IN TRANSACTIONS OF 1000 ROWS;",
        "",
        "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row",
        "CALL {",
        "  WITH row",
        "  MATCH (a:Entity {id: row[':START_ID']})",
        "  MATCH (b:Entity {id: row[':END_ID']})",
    ]

    for edge_type in edge_types:
        lines.extend(
            [
                f"  FOREACH (_ IN CASE WHEN row[':TYPE'] = '{edge_type}' THEN [1] ELSE [] END |",
                f"    MERGE (a)-[r:{edge_type}]->(b)",
                "    SET r.relation = row.relation,",
                "        r.confidence = row.confidence,",
                "        r.weight = toFloat(row['weight:float']),",
                "        r.source_file = row.source_file",
                "  )",
            ]
        )

    lines.extend(
        [
            "} IN TRANSACTIONS OF 1000 ROWS;",
            "",
            "LOAD CSV WITH HEADERS FROM 'file:///hyperedge_members.csv' AS row",
            "CALL {",
            "  WITH row",
            "  MATCH (a:Entity {id: row[':START_ID']})",
            "  MATCH (b:Entity {id: row[':END_ID']})",
            "  MERGE (a)-[r:HAS_MEMBER]->(b)",
            "  SET r.relation = row.relation,",
            "      r.confidence = row.confidence,",
            "      r.weight = toFloat(row['weight:float']),",
            "      r.source_file = row.source_file",
            "} IN TRANSACTIONS OF 1000 ROWS;",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
