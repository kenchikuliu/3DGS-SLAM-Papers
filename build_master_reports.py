#!/usr/bin/env python3
"""
Build consolidated overview reports for the deduped v3 graph.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_year_trends import infer_year

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2" / "final_reviewed_v3" / "deduped"
GRAPH_PATH = BASE_DIR / "graphify_final_reviewed_v3_deduped.json"
MARKDOWN_DIR = ROOT / "extracted_markdown"
INDEX_DIR = BASE_DIR / "indexes"
GENERAL_SUBINDEX_DIR = INDEX_DIR / "general_subindexes"
OUT_DIR = BASE_DIR / "master_reports"
YEAR_TRENDS_DIR = BASE_DIR / "year_trends"

DOMAIN_ORDER = ["SLAM", "Robotics", "General"]
GENERAL_SUBDOMAIN_ORDER = ["Reconstruction", "Rendering", "Enhancement", "Evaluation"]
MANUAL_TITLE_OVERRIDES = {
    "2405.05526.md": "Benchmarking Neural Radiance Fields for Autonomous Robots: An Overview",
    "2601.05738.md": "FeatureSLAM: Feature-Enriched 3D Gaussian Splatting SLAM in Real Time",
    "2506.18678.md": "MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation",
    "2505.09915.md": "Large-Scale Gaussian Splatting SLAM",
    "2510.23988.md": "A Survey on Collaborative SLAM with 3D Gaussian Splatting",
    "2412.03263.md": "NeRF and Gaussian Splatting SLAM in the Wild",
    "2406.16850.md": "From Perfect to Noisy World Simulation: Customizable Embodied Multi-modal Perturbations for SLAM Robustness Benchmarking",
    "2402.08125.md": "Customizable Perturbation Synthesis for Robust SLAM Benchmarking",
    "2509.11574.md": "Gaussian-plus-SDF SLAM: High-fidelity 3D Reconstruction at 150+ fps",
    "2303.00304.md": "Renderable Neural Radiance Map for Visual Navigation",
    "2502.05752.md": "PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map",
    "2503.12572.md": "Deblur Gaussian Splatting SLAM",
    "2511.23221.md": "Robust 3DGS-based SLAM via Adaptive Kernel Smoothing",
    "2309.13240.md": "NeRF-Enhanced Outpainting for Faithful Field-of-View Extrapolation",
    "2409.20276.md": "Active Neural Mapping at Scale",
    "0149_[SLAM-Supplement]_`arXiv`.md": "Globally Consistent RGB-D SLAM with 2D Gaussian Splatting",
    "0053_[SLAM]_STAMICS Splat, Track And Map with Integrated Consistency and Semantics for Dense RGB-.md": "STAMICS: Splat, Track And Map with Integrated Consistency and Semantics for Dense RGB-D SLAM",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graph = load_json(GRAPH_PATH)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_by_id = {node["id"]: node for node in nodes}

    doc_relations = defaultdict(list)
    doc_degree = Counter()
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        rel = edge.get("relation", "")
        if src in node_by_id and node_by_id[src].get("file_type") in {"document", "paper"}:
            doc_relations[src].append(edge)
            if rel != "belongs_to":
                doc_degree[src] += 1
        if tgt in node_by_id and node_by_id[tgt].get("file_type") in {"document", "paper"} and rel != "belongs_to":
            doc_degree[tgt] += 1

    domain_payloads = {domain: load_json(INDEX_DIR / f"{domain.lower()}_index.json") for domain in DOMAIN_ORDER}
    general_sub_payloads = {
        subdomain: load_json(GENERAL_SUBINDEX_DIR / f"{subdomain.lower()}_index.json")
        for subdomain in GENERAL_SUBDOMAIN_ORDER
    }
    year_summary = load_json(YEAR_TRENDS_DIR / "summary.json")
    year_rows = load_json(YEAR_TRENDS_DIR / "year_trends.json")
    domain_year_rows = load_json(YEAR_TRENDS_DIR / "domain_year_trends.json")

    domain_overview_rows = []
    for domain in DOMAIN_ORDER:
        payload = domain_payloads[domain]
        domain_overview_rows.append(
            {
                "domain": domain,
                "paper_count": payload.get("paper_count", 0),
                "top_dataset": first_label(payload.get("top_datasets", [])),
                "top_metric": first_label(payload.get("top_metrics", [])),
                "top_concept": first_label(payload.get("top_concepts", [])),
            }
        )

    general_subdomain_rows = []
    for subdomain in GENERAL_SUBDOMAIN_ORDER:
        payload = general_sub_payloads[subdomain]
        general_subdomain_rows.append(
            {
                "subdomain": subdomain,
                "paper_count": payload.get("paper_count", 0),
                "top_dataset": first_label(payload.get("top_datasets", [])),
                "top_metric": first_label(payload.get("top_metrics", [])),
                "top_concept": first_label(payload.get("top_concepts", [])),
            }
        )

    global_top_rows = build_ranked_rows(
        node_by_id=node_by_id,
        rows=[node for node in nodes if node.get("file_type") in {"document", "paper"}],
        doc_degree=doc_degree,
        limit=50,
    )

    domain_top_rows = []
    for domain in DOMAIN_ORDER:
        papers = domain_payloads[domain].get("papers", [])
        domain_top_rows.extend(
            build_ranked_rows(
                node_by_id=node_by_id,
                rows=[node_by_id[row["id"]] for row in papers if row["id"] in node_by_id],
                doc_degree=doc_degree,
                limit=20,
                domain=domain,
            )
        )

    resolved_titles = build_resolved_titles(global_top_rows, domain_top_rows, node_by_id)
    apply_resolved_titles(global_top_rows, resolved_titles)
    apply_resolved_titles(domain_top_rows, resolved_titles)
    review_queue = build_title_review_queue(global_top_rows, domain_top_rows)

    summary = {
        "graph": str(GRAPH_PATH),
        "document_count": sum(1 for node in nodes if node.get("file_type") in {"document", "paper"}),
        "domain_counts": {row["domain"]: row["paper_count"] for row in domain_overview_rows},
        "general_subdomain_counts": {row["subdomain"]: row["paper_count"] for row in general_subdomain_rows},
        "year_summary": {
            "resolved_count": year_summary.get("year_resolved_count", 0),
            "unknown_count": year_summary.get("year_unknown_count", 0),
            "range": year_summary.get("year_range", [None, None]),
        },
        "resolved_title_count": len(resolved_titles),
        "title_review_queue_count": len(review_queue),
    }

    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "domain_overview.json", domain_overview_rows)
    write_json(OUT_DIR / "general_subdomain_overview.json", general_subdomain_rows)
    write_json(OUT_DIR / "year_overview.json", year_rows)
    write_json(OUT_DIR / "domain_year_overview.json", domain_year_rows)
    write_json(OUT_DIR / "top_papers_global.json", global_top_rows)
    write_json(OUT_DIR / "top_papers_by_domain.json", domain_top_rows)
    write_json(OUT_DIR / "resolved_titles.json", list(resolved_titles.values()))
    write_json(OUT_DIR / "title_review_queue.json", review_queue)

    write_csv(
        OUT_DIR / "domain_overview.csv",
        domain_overview_rows,
        ["domain", "paper_count", "top_dataset", "top_metric", "top_concept"],
    )
    write_csv(
        OUT_DIR / "general_subdomain_overview.csv",
        general_subdomain_rows,
        ["subdomain", "paper_count", "top_dataset", "top_metric", "top_concept"],
    )
    write_csv(
        OUT_DIR / "year_overview.csv",
        year_rows,
        ["year", "paper_count", *DOMAIN_ORDER, "SLAM-Supplement", "Unknown"],
    )
    write_csv(
        OUT_DIR / "domain_year_overview.csv",
        flatten_domain_year_rows(domain_year_rows),
        ["domain", "timeline"],
    )
    write_csv(
        OUT_DIR / "top_papers_global.csv",
        global_top_rows,
        ["rank", "domain", "year", "label", "source_file", "degree", "alias_count"],
    )
    write_csv(
        OUT_DIR / "top_papers_by_domain.csv",
        domain_top_rows,
        ["rank", "domain", "year", "label", "source_file", "degree", "alias_count"],
    )
    write_csv(
        OUT_DIR / "title_review_queue.csv",
        review_queue,
        ["priority", "domain", "rank", "label", "original_label", "source_file", "reason"],
    )

    (OUT_DIR / "master_index.md").write_text(
        render_master_markdown(
            summary,
            domain_overview_rows,
            general_subdomain_rows,
            year_rows,
            domain_year_rows,
            global_top_rows,
            domain_top_rows,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "master_index.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "domain_overview": domain_overview_rows,
                "general_subdomain_overview": general_subdomain_rows,
                "year_overview": year_rows,
                "domain_year_overview": domain_year_rows,
                "top_papers_global": global_top_rows,
                "top_papers_by_domain": domain_top_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False))


def build_ranked_rows(
    node_by_id: dict,
    rows: list[dict],
    doc_degree: Counter,
    limit: int,
    domain: str | None = None,
) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda node: (
            -doc_degree.get(node["id"], 0),
            -len(node.get("alias_ids", [])),
            node.get("label", ""),
        ),
    )[:limit]

    results = []
    for idx, node in enumerate(ranked, start=1):
        results.append(
            {
                "rank": idx,
                "domain": domain or node.get("category", "Unknown"),
                "year": infer_year(node.get("source_file", ""), node.get("label", "")) or "",
                "label": node.get("label", ""),
                "node_id": node.get("id", ""),
                "source_file": node.get("source_file", ""),
                "degree": doc_degree.get(node["id"], 0),
                "alias_count": len(node.get("alias_ids", [])),
            }
        )
    return results


def first_label(rows: list[dict]) -> str:
    return rows[0]["label"] if rows else ""


def build_resolved_titles(global_top_rows: list[dict], domain_top_rows: list[dict], node_by_id: dict) -> dict:
    candidates = {}
    for row in global_top_rows + domain_top_rows:
        label = row.get("label", "")
        source_file = row.get("source_file", "")
        manual = MANUAL_TITLE_OVERRIDES.get(source_file)
        if manual and manual != label:
            candidates[row["node_id"]] = {
                "node_id": row["node_id"],
                "original_label": label,
                "resolved_label": manual,
                "source_file": source_file,
                "resolution_type": "manual",
            }
            continue
        if looks_like_identifier_title(label):
            resolved = MANUAL_TITLE_OVERRIDES.get(source_file) or resolve_title_from_markdown(source_file)
            if resolved and resolved != label:
                candidates[row["node_id"]] = {
                    "node_id": row["node_id"],
                    "original_label": label,
                    "resolved_label": resolved,
                    "source_file": source_file,
                    "resolution_type": "manual" if source_file in MANUAL_TITLE_OVERRIDES else "heuristic",
                }
    return candidates


def apply_resolved_titles(rows: list[dict], resolved_titles: dict) -> None:
    for row in rows:
        resolved = resolved_titles.get(row.get("node_id"))
        if resolved:
            row["original_label"] = row["label"]
            row["label"] = resolved["resolved_label"]


def looks_like_identifier_title(label: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{4}\.[0-9]{5}(v\d+)?(?: \(\d+\))?", label.strip()))


def resolve_title_from_markdown(source_file: str) -> str:
    path = MARKDOWN_DIR / source_file
    if not path.exists():
        return ""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cleaned = []
    for raw in lines[:40]:
        line = raw.strip()
        if not line or line.startswith("<!--"):
            continue
        line = line.lstrip("#").strip()
        if not line:
            continue
        if should_skip_title_line(line):
            continue
        cleaned.append(line)

    if not cleaned:
        return ""

    title_parts = []
    for line in cleaned:
        if looks_like_author_line(line) or looks_like_section_heading(line):
            break
        if re.fullmatch(r"\d+", line):
            continue
        title_parts.append(line)
        joined = " ".join(title_parts)
        if len(title_parts) >= 2 and len(joined) >= 24 and not joined.endswith(":"):
            break
        if len(title_parts) == 1 and len(joined) >= 72 and not joined.endswith(":"):
            break

    title = normalize_title(" ".join(title_parts))
    if title and not looks_like_identifier_title(title):
        return title
    return ""


def looks_like_author_line(line: str) -> bool:
    lower = line.lower()
    tokens = lower.split()
    if "@" in line:
        return True
    if any(term in lower for term in ["university", "department", "school of", "institute of", "laboratory", "college", "research center"]):
        return True
    if "abstract" in lower or "introduction" in lower:
        return True
    if len(tokens) >= 2 and looks_like_name_sequence(line):
        return True
    if len(tokens) > 3 and sum(1 for token in tokens if token[:1].isupper()) >= max(2, len(tokens) // 2):
        return any(ch.isdigit() for ch in line) or "," in line
    return False


def looks_like_section_heading(line: str) -> bool:
    upper = line.upper()
    return upper.startswith(("ABSTRACT", "I.", "II.", "III.", "1.", "2.")) or "INTRODUCTION" in upper


def should_skip_title_line(line: str) -> bool:
    lower = line.lower()
    if lower.startswith("arxiv:") or "[cs." in lower or re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", lower):
        return True
    journal_terms = [
        "ieee transactions", "transactions on", "conference on", "proceedings of",
        "cvpr", "iccv", "eccv", "rss", "iros", "ral", "neurips", "tpami",
    ]
    if any(term in lower for term in journal_terms):
        return True
    letters = [ch for ch in line if ch.isalpha()]
    if letters and sum(1 for ch in letters if ch.isupper()) / len(letters) > 0.8 and len(line) > 20:
        return True
    return False


def looks_like_name_sequence(line: str) -> bool:
    cleaned = re.sub(r"[\d\*\u2020,]", " ", line)
    tokens = [token for token in cleaned.split() if token]
    if not 2 <= len(tokens) <= 14:
        return False
    stopwords = {"for", "with", "and", "using", "from", "the", "of", "in", "on", "to", "via", "a", "an"}
    if any(token.lower() in stopwords for token in tokens):
        return False
    capitalized = 0
    for token in tokens:
        stripped = token.strip("-")
        if stripped[:1].isupper() and stripped[1:].islower():
            capitalized += 1
    return capitalized >= max(2, len(tokens) - 1)


def normalize_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text[:160].strip()


def build_title_review_queue(global_top_rows: list[dict], domain_top_rows: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for row in global_top_rows + domain_top_rows:
        key = row.get("node_id")
        if key in seen:
            continue
        seen.add(key)
        reason = suspicious_title_reason(row.get("label", ""))
        if not reason:
            continue
        priority = "high" if row.get("rank", 999) <= 20 else "medium"
        rows.append(
            {
                "priority": priority,
                "domain": row.get("domain", ""),
                "rank": row.get("rank", 0),
                "label": row.get("label", ""),
                "original_label": row.get("original_label", ""),
                "source_file": row.get("source_file", ""),
                "reason": reason,
            }
        )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (priority_order.get(row["priority"], 9), row["domain"], row["rank"], row["source_file"]))
    return rows


def suspicious_title_reason(label: str) -> str:
    if not label:
        return "empty"
    if label in {"Overview"}:
        return "too_short"
    if len(label) < 24:
        return "too_short"
    if "@" in label:
        return "contains_email"
    if label.count(",") >= 3:
        return "looks_like_author_list"
    if re.search(r"\b[A-Z][a-z]+\d", label):
        return "contains_affiliation_markers"
    if label.upper().startswith(("IEEE ", "CVPR ", "ICCV ", "ECCV ")):
        return "journal_or_venue_header"
    return ""


def flatten_domain_year_rows(domain_year_rows: dict) -> list[dict]:
    rows = []
    for domain, entries in domain_year_rows.items():
        timeline = " | ".join(f"{entry['year']}:{entry['paper_count']}" for entry in entries)
        rows.append({"domain": domain, "timeline": timeline})
    return rows


def render_master_markdown(
    summary: dict,
    domain_rows: list[dict],
    subdomain_rows: list[dict],
    year_rows: list[dict],
    domain_year_rows: dict,
    global_top_rows: list[dict],
    domain_top_rows: list[dict],
) -> str:
    lines = [
        "# Master Index",
        "",
        f"- document count: {summary['document_count']}",
        f"- year resolved count: {summary['year_summary']['resolved_count']}",
        f"- year unknown count: {summary['year_summary']['unknown_count']}",
        f"- year range: {summary['year_summary']['range'][0]} -> {summary['year_summary']['range'][1]}",
        "",
        "## Domain Overview",
        "",
    ]
    for row in domain_rows:
        lines.append(
            f"- {row['domain']}: {row['paper_count']} papers | top dataset {row['top_dataset']} | "
            f"top metric {row['top_metric']} | top concept {row['top_concept']}"
        )

    lines.extend(["", "## General Subdomains", ""])
    for row in subdomain_rows:
        lines.append(
            f"- {row['subdomain']}: {row['paper_count']} papers | top dataset {row['top_dataset']} | "
            f"top metric {row['top_metric']} | top concept {row['top_concept']}"
        )

    lines.extend(["", "## Year Overview", ""])
    for row in year_rows:
        lines.append(
            f"- {row['year']}: {row['paper_count']} | SLAM {row['SLAM']} | Robotics {row['Robotics']} | "
            f"General {row['General']} | SLAM-Supplement {row['SLAM-Supplement']} | Unknown {row['Unknown']}"
        )

    lines.extend(["", "## Domain Year Timelines", ""])
    for domain in ["SLAM", "Robotics", "General", "SLAM-Supplement", "Unknown"]:
        entries = domain_year_rows.get(domain, [])
        timeline = " | ".join(f"{entry['year']}:{entry['paper_count']}" for entry in entries) or "-"
        lines.append(f"- {domain}: {timeline}")

    lines.extend(["", "## Top Papers Global", ""])
    for row in global_top_rows[:20]:
        lines.append(
            f"- #{row['rank']} [{row['domain']}] {row['label']} | degree {row['degree']} | {row['source_file']}"
        )

    lines.extend(["", "## Top Papers By Domain", ""])
    grouped = defaultdict(list)
    for row in domain_top_rows:
        grouped[row["domain"]].append(row)
    for domain in DOMAIN_ORDER:
        lines.extend(["", f"### {domain}", ""])
        for row in grouped.get(domain, [])[:10]:
            lines.append(
                f"- #{row['rank']} {row['label']} | degree {row['degree']} | {row['source_file']}"
            )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
