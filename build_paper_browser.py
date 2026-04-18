#!/usr/bin/env python3
"""
Build a local HTML browser for deduped papers with direct links to markdown files.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_year_trends import infer_year

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2" / "final_reviewed_v3" / "deduped"
GRAPH_PATH = BASE_DIR / "graphify_final_reviewed_v3_deduped.json"
MASTER_DIR = BASE_DIR / "master_reports"
OUTPUT_JSON = MASTER_DIR / "paper_browser.json"
OUTPUT_HTML = MASTER_DIR / "paper_browser.html"
MARKDOWN_DIR = ROOT / "extracted_markdown"


def main() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    rows = build_rows(graph)
    payload = {
        "summary": {
            "paper_count": len(rows),
            "domains": domain_counts(rows),
        },
        "papers": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"json": str(OUTPUT_JSON), "html": str(OUTPUT_HTML), "paper_count": len(rows)}, ensure_ascii=False))


def build_rows(graph: dict) -> list[dict]:
    rows = []
    for node in graph.get("nodes", []):
        if node.get("file_type") not in {"document", "paper"}:
            continue
        source_file = node.get("source_file", "")
        md_path = MARKDOWN_DIR / source_file
        preview_text = load_preview(md_path)
        rows.append(
            {
                "id": node.get("id", ""),
                "title": node.get("label", ""),
                "domain": node.get("category", "Unknown"),
                "year": infer_year(source_file, node.get("label", "")) or "",
                "source_file": source_file,
                "markdown_path": str(md_path),
                "markdown_url": md_path.as_uri() if md_path.exists() else "",
                "alias_count": len(node.get("alias_ids", [])),
                "preview_text": preview_text,
            }
        )
    rows.sort(key=lambda row: (row["domain"], str(row["year"]), row["title"].lower(), row["source_file"].lower()))
    return rows


def domain_counts(rows: list[dict]) -> dict:
    counts = {}
    for row in rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    return counts


def load_preview(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("<!-- page"):
            continue
        lines.append(line)
    preview = "\n".join(lines).strip()
    return preview[:limit]


def render_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3DGS-SLAM Paper Browser</title>
  <style>
    :root {{
      --bg: #f5f7f8;
      --surface: #ffffff;
      --line: #d2dde1;
      --text: #132025;
      --muted: #55666d;
      --accent: #0d7a6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .panel + .panel {{ margin-top: 16px; }}
    .layout {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 16px;
      align-items: start;
      margin-top: 16px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 10px;
      margin: 14px 0;
    }}
    input, select {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .chip {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #eef3f4;
      font-size: 13px;
    }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .preview {{
      white-space: pre-wrap;
      font-family: Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
      max-height: 70vh;
      overflow: auto;
      background: #fbfcfc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    tr.active {{
      background: #eef6f4;
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1 style="margin:0 0 8px;">3DGS-SLAM Paper Browser</h1>
      <div class="small">Local browser for deduped papers with direct markdown links.</div>
      <div class="stats" id="stats"></div>
    </div>
    <div class="layout">
      <div class="panel">
        <div class="controls">
          <input id="search" type="text" placeholder="Search title or source file">
          <select id="domainFilter"><option value="all">All domains</option></select>
          <select id="yearFilter"><option value="all">All years</option></select>
        </div>
        <table id="paperTable"></table>
      </div>
      <div class="panel">
        <h2 style="margin:0 0 12px;">Preview</h2>
        <div id="previewMeta" class="small" style="margin-bottom:12px;"></div>
        <div id="previewBody" class="preview">Select a paper to preview markdown.</div>
      </div>
    </div>
  </div>
  <script>
    const payload = {data};
    const papers = payload.papers || [];
    const stats = document.getElementById('stats');
    const table = document.getElementById('paperTable');
    const search = document.getElementById('search');
    const domainFilter = document.getElementById('domainFilter');
    const yearFilter = document.getElementById('yearFilter');
    const previewMeta = document.getElementById('previewMeta');
    const previewBody = document.getElementById('previewBody');
    let selectedId = papers.length ? papers[0].id : '';

    applyUrlParams();
    renderStats();
    populateFilter();
    renderTable();
    renderPreview(papers.find(row => row.id === selectedId) || null);

    search.addEventListener('input', renderTable);
    domainFilter.addEventListener('change', renderTable);
    yearFilter.addEventListener('change', renderTable);

    function renderStats() {{
      const chips = [`<div class="chip">papers: ${{payload.summary.paper_count || 0}}</div>`];
      for (const [domain, count] of Object.entries(payload.summary.domains || {{}})) {{
        chips.push(`<div class="chip">${{escapeHtml(domain)}}: ${{count}}</div>`);
      }}
      stats.innerHTML = chips.join('');
    }}

    function populateFilter() {{
      const domains = ['all', ...new Set(papers.map(row => row.domain))];
      const years = ['all', ...new Set(papers.map(row => row.year).filter(Boolean))].sort((a, b) => a === 'all' ? -1 : Number(a) - Number(b));
      domainFilter.innerHTML = domains.map(domain => {{
        const label = domain === 'all' ? 'All domains' : domain;
        return `<option value="${{domain}}">${{label}}</option>`;
      }}).join('');
      yearFilter.innerHTML = years.map(year => {{
        const label = year === 'all' ? 'All years' : year;
        return `<option value="${{year}}">${{label}}</option>`;
      }}).join('');
      if (window.initialDomain && domains.includes(window.initialDomain)) domainFilter.value = window.initialDomain;
      if (window.initialYear && years.includes(window.initialYear)) yearFilter.value = window.initialYear;
    }}

    function renderTable() {{
      const term = search.value.trim().toLowerCase();
      const domain = domainFilter.value;
      const year = yearFilter.value;
      const rows = papers.filter(row => {{
        const matchDomain = domain === 'all' || row.domain === domain;
        const matchYear = year === 'all' || String(row.year) === year;
        const text = `${{row.title}} ${{row.source_file}}`.toLowerCase();
        return matchDomain && matchYear && (!term || text.includes(term));
      }});
      if (!rows.some(row => row.id === selectedId) && rows.length) {{
        selectedId = rows[0].id;
      }}
      table.innerHTML = `
        <thead>
          <tr>
            <th>Domain</th>
            <th>Year</th>
            <th>Title</th>
            <th>Markdown</th>
            <th>Aliases</th>
          </tr>
        </thead>
        <tbody>
          ${{rows.map(row => `
            <tr class="${{row.id === selectedId ? 'active' : ''}}" data-id="${{row.id}}">
              <td>${{escapeHtml(row.domain)}}</td>
              <td>${{escapeHtml(String(row.year || ''))}}</td>
              <td>
                <div>${{escapeHtml(row.title)}}</div>
                <div class="small">${{escapeHtml(row.source_file)}}</div>
              </td>
              <td>${{row.markdown_url ? `<a href="${{row.markdown_url}}">open markdown</a>` : '<span class="small">missing</span>'}}</td>
              <td>${{row.alias_count}}</td>
            </tr>
          `).join('')}}
        </tbody>`;
      for (const tr of table.querySelectorAll('tbody tr')) {{
        tr.addEventListener('click', () => {{
          const row = papers.find(item => item.id === tr.dataset.id);
          if (!row) return;
          selectedId = row.id;
          renderTable();
          renderPreview(row);
        }});
      }}
    }}

    function renderPreview(row) {{
      if (!row) {{
        previewMeta.textContent = '';
        previewBody.textContent = 'No paper selected.';
        return;
      }}
      previewMeta.innerHTML = `
        <div><strong>${{escapeHtml(row.title)}}</strong></div>
        <div>${{escapeHtml(row.domain)}} | year ${{escapeHtml(String(row.year || '-'))}} | ${{escapeHtml(row.source_file)}} | aliases ${{row.alias_count}}</div>
        <div>${{row.markdown_url ? `<a href="${{row.markdown_url}}">open full markdown</a>` : ''}}</div>`;
      previewBody.textContent = row.preview_text || 'No preview available.';
    }}

    function applyUrlParams() {{
      const params = new URLSearchParams(window.location.search);
      window.initialDomain = params.get('domain') || '';
      window.initialYear = params.get('year') || '';
      const initialSearch = params.get('search') || '';
      if (initialSearch) search.value = initialSearch;
    }}

    function escapeHtml(value) {{
      return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
