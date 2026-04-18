#!/usr/bin/env python3
"""
Build a shortlist from the low-priority review queue so manual review can focus
on the most ambiguous or highest-yield items first.
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

KEYWORDS = {
    "slam": 3,
    "localization": 3,
    "mapping": 3,
    "odometry": 3,
    "relocalization": 3,
    "robot": 2,
    "robotics": 2,
    "navigation": 2,
    "multi-agent": 2,
    "multi agent": 2,
    "active": 1,
    "lidar": 1,
    "visual-inertial": 1,
    "imu": 1,
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def first_nonempty_lines(text: str, limit: int = 8) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line.startswith("<!-- page"):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def keyword_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    for keyword, weight in KEYWORDS.items():
        if keyword in lowered:
            score += weight
    return score


def main() -> None:
    rows = []
    with open(QUEUE_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("review_priority") == "low":
                rows.append(row)

    enriched = []
    for row in rows:
        text = read_text(MD_DIR / row["source_file"])
        snippet = " ".join(first_nonempty_lines(text))[:1000]
        score = keyword_score(f"{row['label']} {snippet}")
        margin = int(row.get("confidence_margin", "999") or 999)
        enriched.append(
            {
                **row,
                "keyword_score": score,
                "snippet": snippet,
                "sort_key": (margin, -score, row["source_file"]),
            }
        )

    enriched.sort(key=lambda item: item["sort_key"])
    shortlist = enriched[:25]

    md_lines = ["# Low Priority Shortlist", "", "## Top 25", ""]
    for idx, row in enumerate(shortlist, start=1):
        md_lines.extend(
            [
                f"### {idx}. {row['label']}",
                "",
                f"- source file: `{row['source_file']}`",
                f"- suggested category: `{row['new_category']}`",
                f"- confidence margin: `{row['confidence_margin']}`",
                f"- keyword score: `{row['keyword_score']}`",
                f"- rationale: `{row['rationale']}`",
                "",
                row["snippet"] or "_No snippet extracted._",
                "",
            ]
        )

    payload = {
        "total_low_priority_count": len(enriched),
        "shortlist_count": len(shortlist),
        "items": shortlist,
    }
    (REVIEW_DIR / "low_priority_shortlist.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REVIEW_DIR / "low_priority_shortlist.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"total_low_priority_count": len(enriched), "shortlist_count": len(shortlist)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
