#!/usr/bin/env python3
"""
Build a readable review packet for category assignments that need manual review.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
MD_DIR = ROOT / "extracted_markdown"
REVIEW_DIR = ROOT / "graphify-out" / "filtered" / "recategorized"
QUEUE_CSV = REVIEW_DIR / "unknown_category_review_queue.csv"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def first_nonempty_lines(text: str, limit: int = 12) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if line.startswith("<!-- page"):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def extract_abstractish(lines: list[str]) -> str:
    if not lines:
        return ""
    merged = " ".join(lines[:8])
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged[:1200]


def main() -> None:
    rows = []
    with open(QUEUE_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    high_rows = [row for row in rows if row.get("review_priority") == "high"]
    medium_rows = [row for row in rows if row.get("review_priority") == "medium"]

    packet = {
        "high_priority_count": len(high_rows),
        "medium_priority_count": len(medium_rows),
        "items": [],
    }

    md_lines = ["# Category Review Packet", "", "## High Priority", ""]

    for idx, row in enumerate(high_rows, start=1):
        source_file = row["source_file"]
        md_path = MD_DIR / source_file
        text = read_text(md_path)
        lines = first_nonempty_lines(text)
        snippet = extract_abstractish(lines)

        item = {
            "rank": idx,
            "id": row["id"],
            "label": row["label"],
            "source_file": source_file,
            "new_category": row["new_category"],
            "rationale": row["rationale"],
            "confidence_margin": row["confidence_margin"],
            "review_priority": row["review_priority"],
            "snippet": snippet,
        }
        packet["items"].append(item)

        md_lines.extend(
            [
                f"### {idx}. {row['label']}",
                "",
                f"- source file: `{source_file}`",
                f"- suggested category: `{row['new_category']}`",
                f"- confidence margin: `{row['confidence_margin']}`",
                f"- rationale: `{row['rationale']}`",
                "",
                snippet if snippet else "_No snippet extracted._",
                "",
            ]
        )

    md_lines.extend(["## Medium Priority", ""])
    for row in medium_rows[:20]:
        md_lines.append(
            f"- {row['label']} | suggested `{row['new_category']}` | margin `{row['confidence_margin']}` | {row['source_file']}"
        )

    (REVIEW_DIR / "category_review_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REVIEW_DIR / "category_review_packet.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"high_priority_count": len(high_rows), "medium_priority_count": len(medium_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
