#!/usr/bin/env python3
"""
Build year trend reports for the deduped graph.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2" / "final_reviewed_v3" / "deduped"
GRAPH_PATH = BASE_DIR / "graphify_final_reviewed_v3_deduped.json"
OUT_DIR = BASE_DIR / "year_trends"
MARKDOWN_DIR = ROOT / "extracted_markdown"

DOMAIN_ORDER = ["SLAM", "Robotics", "General", "SLAM-Supplement", "Unknown"]

MANUAL_YEAR_OVERRIDES = {
    "0047_[SLAM]_FGS-SLAM Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map.md": 2025,
    "0053_[SLAM]_STAMICS Splat, Track And Map with Integrated Consistency and Semantics for Dense RGB-.md": 2025,
    "0058_[SLAM]_FGO-SLAM Online Co-Design Optimization of Sensing, Computing, and Control for Gaussi.md": 2025,
    "0073_[Robotics]_Ditto Building Digital Twins of Articulated Objects from Interacti.md": 2022,
    "0069_[Robotics]_NeuralGrasps Learning Implicit Representations for Grasping of Unknown Objects.md": 2023,
    "0071_[Robotics]_ObjectFolder A Dataset of Objects with Implicit Visual, Auditory, and Tactile Re.md": 2021,
    "0088_[Robotics]_NeRFlow Neural Radiance Flow for 4D View Synthesis and Video Process.md": 2021,
    "0075_[Robotics]_CLIP-Fields Open-label semantic navigation with pre-trained VLMs and language gro.md": 2022,
    "0079_[Robotics]_GaussNav Gaussian Splatting for Visual Navigation.md": 2024,
    "0149_[SLAM-Supplement]_`arXiv`.md": 2025,
    "0279_[General]_Three-dimensional Damage Visualization of Civil Structures v.md": 2026,
    "0316_[General]_LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D.md": 2026,
    "0375_[General]_GaussianPlant Structure-aligned Gaussian Splatting for 3D R.md": 2025,
    "ssrn-4514612.md": 2024,
}

MANUAL_TITLE_YEAR_OVERRIDES = {
    "FGS-SLAM": 2025,
    "STAMICS": 2025,
    "FGO-SLAM": 2025,
    "Ditto Building Digital Twins of Articulated Objects from Interacti": 2022,
    "NeuralGrasps": 2023,
    "ObjectFolder 1.0": 2021,
    "NeRFlow Neural Radiance Flow for 4D View Synthesis and Video Process": 2021,
    "CLIP-Fields": 2022,
    "GaussNav": 2024,
    "`arXiv`": 2025,
    "Three-dimensional Damage Visualization of Civil Structures v": 2026,
    "LL-GaussianMap Zero-shot Low-Light Image Enhancement via 2D": 2026,
    "GaussianPlant Structure-aligned Gaussian Splatting for 3D R": 2025,
    "ssrn-4514612": 2024,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

    rows = []
    unknown_rows = []
    for node in graph.get("nodes", []):
        if node.get("file_type") not in {"document", "paper"}:
            continue
        year = infer_year(node.get("source_file", ""), node.get("label", ""))
        row = {
            "title": node.get("label", ""),
            "domain": node.get("category", "Unknown"),
            "source_file": node.get("source_file", ""),
            "year": year or "",
        }
        if year:
            rows.append(row)
        else:
            unknown_rows.append(row)

    year_counts = Counter(row["year"] for row in rows)
    year_domain_counts = defaultdict(Counter)
    for row in rows:
        year_domain_counts[row["year"]][row["domain"]] += 1

    trend_rows = []
    for year in sorted(year_counts):
        payload = {
            "year": year,
            "paper_count": year_counts[year],
        }
        for domain in DOMAIN_ORDER:
            payload[domain] = year_domain_counts[year].get(domain, 0)
        trend_rows.append(payload)

    summary = {
        "graph": str(GRAPH_PATH),
        "document_count": len(rows) + len(unknown_rows),
        "year_resolved_count": len(rows),
        "year_unknown_count": len(unknown_rows),
        "year_range": [min(year_counts) if year_counts else None, max(year_counts) if year_counts else None],
    }

    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "year_trends.json", trend_rows)
    write_json(OUT_DIR / "year_unknown.json", unknown_rows)
    write_json(OUT_DIR / "domain_year_trends.json", build_domain_year_trends(trend_rows))
    write_json(OUT_DIR / "year_unknown_review_queue.json", build_year_unknown_review_queue(unknown_rows))
    write_csv(
        OUT_DIR / "year_trends.csv",
        trend_rows,
        ["year", "paper_count", *DOMAIN_ORDER],
    )
    write_csv(
        OUT_DIR / "year_unknown.csv",
        unknown_rows,
        ["domain", "title", "source_file"],
    )
    write_csv(
        OUT_DIR / "year_unknown_review_queue.csv",
        build_year_unknown_review_queue(unknown_rows),
        ["priority", "domain", "title", "source_file"],
    )
    (OUT_DIR / "year_trends.md").write_text(render_markdown(summary, trend_rows), encoding="utf-8")
    write_domain_csvs(trend_rows)
    (OUT_DIR / "year_trends.html").write_text(render_html(summary, trend_rows, unknown_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def infer_year(source_file: str, title: str) -> int | None:
    if source_file in MANUAL_YEAR_OVERRIDES:
        return MANUAL_YEAR_OVERRIDES[source_file]
    if title in MANUAL_TITLE_YEAR_OVERRIDES:
        return MANUAL_TITLE_YEAR_OVERRIDES[title]
    if match := re.match(r"^(\d{2})\d{2}\.\d{5}", source_file):
        return 2000 + int(match.group(1))
    if match := re.search(r"\b(20\d{2})\b", source_file):
        year = int(match.group(1))
        if 2015 <= year <= 2030:
            return year
    if match := re.search(r"\b(20\d{2})\b", title):
        year = int(match.group(1))
        if 2015 <= year <= 2030:
            return year
    if inferred := infer_year_from_markdown(source_file):
        return inferred
    return None


def infer_year_from_markdown(source_file: str) -> int | None:
    path = resolve_markdown_path(source_file)
    if not path.exists():
        return None

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:500]
    except OSError:
        return None

    candidates = Counter()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if match := re.search(r"\barxiv:(\d{2})\d{2}\.\d{4,5}(?:v\d+)?\b", lower):
            year = 2000 + int(match.group(1))
            if 2015 <= year <= 2030:
                candidates[year] += 9 if idx < 120 else 6
        if match := re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+(20\d{2}))\b", line):
            year = int(match.group(2))
            if 2015 <= year <= 2030:
                candidates[year] += 6 if idx < 80 else 3
        if match := re.search(r"\b(?:accepted|published|appeared|proceedings|journal|conference|article|preprint)\D{0,30}(20\d{2})\b", lower):
            year = int(match.group(1))
            if 2015 <= year <= 2030:
                candidates[year] += 5
        for match in re.finditer(r"\b(20\d{2})\b", line):
            year = int(match.group(1))
            if not 2015 <= year <= 2030:
                continue
            score = 1
            if idx < 25:
                score += 2
            elif idx < 60:
                score += 1
            context_terms = [
                "arxiv", "cvpr", "iccv", "eccv", "neurips", "icra", "iros", "rss",
                "ieee", "transactions", "conference", "journal", "published",
                "accepted", "preprint", "article", "letter", "doi", "copyright",
            ]
            if any(term in lower for term in context_terms):
                score += 3
            if re.search(rf"(©|\bcopyright\b).*{year}", lower):
                score += 4
            if re.search(rf"\b{year}\b.*(ieee|cvpr|iccv|eccv|icra|iros|rss|arxiv|conference|journal)", lower):
                score += 3
            if "reference" in lower or "bibliography" in lower:
                score -= 3
            candidates[year] += score

    if not candidates:
        return None
    year, score = max(candidates.items(), key=lambda item: (item[1], item[0]))
    return year if score >= 4 else None


def resolve_markdown_path(source_file: str) -> Path:
    direct_path = MARKDOWN_DIR / source_file
    if direct_path.exists():
        return direct_path

    source_stem = Path(source_file).stem.lower()
    source_prefix = source_stem.split(" ")[0]
    candidates = []
    for path in MARKDOWN_DIR.glob("*.md"):
        stem = path.stem.lower()
        score = 0
        if stem == source_stem:
            score += 100
        if stem.startswith(source_prefix):
            score += 20
        if source_stem[:32] and stem.startswith(source_stem[:32]):
            score += 10
        shared_tokens = set(re.findall(r"[a-z0-9]+", source_stem)) & set(re.findall(r"[a-z0-9]+", stem))
        score += len(shared_tokens)
        if score:
            candidates.append((score, path))

    if not candidates:
        return direct_path
    return max(candidates, key=lambda item: (item[0], len(item[1].name)))[1]


def build_domain_year_trends(trend_rows: list[dict]) -> dict:
    payload = {}
    for domain in DOMAIN_ORDER:
        payload[domain] = [
            {"year": row["year"], "paper_count": row.get(domain, 0)}
            for row in trend_rows
            if row.get(domain, 0)
        ]
    return payload


def write_domain_csvs(trend_rows: list[dict]) -> None:
    for domain in DOMAIN_ORDER:
        rows = [
            {"year": row["year"], "paper_count": row.get(domain, 0)}
            for row in trend_rows
            if row.get(domain, 0)
        ]
        write_csv(
            OUT_DIR / f"{domain.lower().replace('-', '_')}_year_trends.csv",
            rows,
            ["year", "paper_count"],
        )


def build_year_unknown_review_queue(unknown_rows: list[dict]) -> list[dict]:
    def priority(row: dict) -> str:
        text = f"{row['title']} {row['source_file']}".lower()
        hot_terms = [
            "slam", "robot", "navigation", "mapping", "localization", "gaussian",
            "survey", "benchmark", "planner", "reconstruction",
        ]
        if any(term in text for term in hot_terms):
            return "high"
        return "medium"

    rows = [
        {
            "priority": priority(row),
            "domain": row["domain"],
            "title": row["title"],
            "source_file": row["source_file"],
        }
        for row in unknown_rows
    ]
    order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (order[row["priority"]], row["domain"], row["source_file"]))
    return rows


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict, trend_rows: list[dict]) -> str:
    lines = [
        "# Year Trends",
        "",
        f"- document count: {summary['document_count']}",
        f"- year resolved count: {summary['year_resolved_count']}",
        f"- year unknown count: {summary['year_unknown_count']}",
        "",
        "## Counts By Year",
        "",
    ]
    for row in trend_rows:
        lines.append(
            f"- {row['year']}: {row['paper_count']} | "
            f"SLAM {row['SLAM']} | Robotics {row['Robotics']} | General {row['General']} | "
            f"SLAM-Supplement {row['SLAM-Supplement']} | Unknown {row['Unknown']}"
        )
    lines.append("")
    return "\n".join(lines)


def render_html(summary: dict, trend_rows: list[dict], unknown_rows: list[dict]) -> str:
    data = json.dumps(
        {
            "summary": summary,
            "trends": trend_rows,
            "unknown": unknown_rows,
            "domain_trends": build_domain_year_trends(trend_rows),
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3DGS-SLAM Year Trends</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7f8; color: #132025; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .panel {{ background: #fff; border: 1px solid #d2dde1; border-radius: 8px; padding: 16px; }}
    .panel + .panel {{ margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #d2dde1; text-align: left; vertical-align: top; }}
    th {{ color: #55666d; }}
    .stats {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
    .chip {{ padding: 6px 10px; border: 1px solid #d2dde1; border-radius: 8px; background: #eef3f4; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1 style="margin:0 0 8px;">3DGS-SLAM Year Trends</h1>
      <div class="stats" id="stats"></div>
    </div>
    <div class="panel">
      <h2 style="margin:0 0 12px;">Counts By Year</h2>
      <table id="trendTable"></table>
    </div>
    <div class="panel">
      <h2 style="margin:0 0 12px;">Domain Timelines</h2>
      <table id="domainTrendTable"></table>
    </div>
    <div class="panel">
      <h2 style="margin:0 0 12px;">Year Unknown</h2>
      <table id="unknownTable"></table>
    </div>
  </div>
  <script>
    const payload = {data};
    const stats = document.getElementById('stats');
    const trendTable = document.getElementById('trendTable');
    const domainTrendTable = document.getElementById('domainTrendTable');
    const unknownTable = document.getElementById('unknownTable');
    stats.innerHTML = [
      ['documents', payload.summary.document_count],
      ['year resolved', payload.summary.year_resolved_count],
      ['year unknown', payload.summary.year_unknown_count],
      ['range', `${{payload.summary.year_range[0] || '-'}} to ${{payload.summary.year_range[1] || '-'}}`]
    ].map(([label, value]) => `<div class="chip">${{label}}: ${{value}}</div>`).join('');
    trendTable.innerHTML = `
      <thead><tr><th>Year</th><th>Total</th><th>SLAM</th><th>Robotics</th><th>General</th><th>SLAM-Supplement</th><th>Unknown</th></tr></thead>
      <tbody>${{payload.trends.map(row => `<tr><td>${{row.year}}</td><td>${{row.paper_count}}</td><td>${{row.SLAM}}</td><td>${{row.Robotics}}</td><td>${{row.General}}</td><td>${{row['SLAM-Supplement']}}</td><td>${{row.Unknown}}</td></tr>`).join('')}}</tbody>`;
    domainTrendTable.innerHTML = `
      <thead><tr><th>Domain</th><th>Timeline</th></tr></thead>
      <tbody>${{Object.entries(payload.domain_trends).map(([domain, rows]) => `<tr><td>${{escapeHtml(domain)}}</td><td>${{rows.map(row => `${{row.year}}:${{row.paper_count}}`).join(' | ')}}</td></tr>`).join('')}}</tbody>`;
    unknownTable.innerHTML = `
      <thead><tr><th>Domain</th><th>Title</th><th>Source</th></tr></thead>
      <tbody>${{payload.unknown.map(row => `<tr><td>${{escapeHtml(row.domain)}}</td><td>${{escapeHtml(row.title)}}</td><td>${{escapeHtml(row.source_file)}}</td></tr>`).join('')}}</tbody>`;
    function escapeHtml(value) {{
      return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
    }}
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
