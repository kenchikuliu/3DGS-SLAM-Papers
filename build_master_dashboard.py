#!/usr/bin/env python3
"""
Build a single-file HTML dashboard for the deduped master reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
BASE_DIR = ROOT / "graphify-out" / "filtered" / "recategorized" / "final_reviewed" / "final_reviewed_v2" / "final_reviewed_v3" / "deduped"
MASTER_DIR = BASE_DIR / "master_reports"
INPUT_JSON = MASTER_DIR / "master_index.json"
OUTPUT_HTML = MASTER_DIR / "master_dashboard.html"
MARKDOWN_DIR = ROOT / "extracted_markdown"


def main() -> None:
    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    payload["paper_browser_url"] = (MASTER_DIR / "paper_browser.html").as_uri()
    payload["year_trends_url"] = (BASE_DIR / "year_trends" / "year_trends.html").as_uri()
    for bucket in ("top_papers_global", "top_papers_by_domain"):
        for row in payload.get(bucket, []):
            md_path = MARKDOWN_DIR / row.get("source_file", "")
            row["markdown_url"] = md_path.as_uri() if md_path.exists() else ""
            browser_params = []
            if row.get("domain"):
                browser_params.append(f"domain={row['domain']}")
            if row.get("year"):
                browser_params.append(f"year={row['year']}")
            if row.get("label"):
                browser_params.append(f"search={row['label']}")
            encoded_params = {}
            for entry in browser_params:
                key, value = entry.split("=", 1)
                encoded_params[key] = value
            row["paper_browser_row_url"] = payload["paper_browser_url"] + ("?" + urlencode(encoded_params) if encoded_params else "")
    OUTPUT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"input": str(INPUT_JSON), "output": str(OUTPUT_HTML)}, ensure_ascii=False))


def render_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3DGS-SLAM Master Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f8;
      --surface: #ffffff;
      --surface-2: #eef3f4;
      --text: #132025;
      --muted: #55666d;
      --line: #d2dde1;
      --accent: #0d7a6b;
      --accent-2: #ad3f2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ margin: 0; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0 24px;
    }}
    .stat {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .stat .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .panel + .panel {{ margin-top: 16px; }}
    .nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .nav a {{
      display: inline-block;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      color: var(--text);
      text-decoration: none;
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
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
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
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .chip {{
      background: var(--surface-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1>3DGS-SLAM Master Dashboard</h1>
      <p>Deduped corpus overview, domain summary, and paper rankings.</p>
      <div class="nav">
        <a id="paperBrowserLink" href="#">Open Paper Browser</a>
        <a id="yearTrendsLink" href="#">Open Year Trends</a>
      </div>
      <div class="stats" id="stats"></div>
    </div>

    <div class="grid" style="margin-top: 16px;">
      <div>
        <div class="panel">
          <h2>Domain Overview</h2>
          <table id="domainTable"></table>
        </div>
        <div class="panel">
          <h2>General Subdomains</h2>
          <table id="subdomainTable"></table>
        </div>
        <div class="panel">
          <h2>Year Overview</h2>
          <table id="yearTable"></table>
        </div>
        <div class="panel">
          <h2>Domain Timelines</h2>
          <table id="domainYearTable"></table>
        </div>
      </div>

      <div>
        <div class="panel">
          <h2>Paper Rankings</h2>
          <div class="controls">
            <input id="search" type="text" placeholder="Search title or file">
            <select id="domainFilter">
              <option value="all">All domains</option>
            </select>
            <select id="scopeFilter">
              <option value="global">Global top 50</option>
              <option value="domain">Per-domain top 20</option>
            </select>
            <select id="yearFilter">
              <option value="all">All years</option>
            </select>
          </div>
          <table id="paperTable"></table>
        </div>
      </div>
    </div>
  </div>
  <script>
    const payload = {data};
    const stats = document.getElementById('stats');
    const domainTable = document.getElementById('domainTable');
    const subdomainTable = document.getElementById('subdomainTable');
    const yearTable = document.getElementById('yearTable');
    const domainYearTable = document.getElementById('domainYearTable');
    const paperTable = document.getElementById('paperTable');
    const search = document.getElementById('search');
    const domainFilter = document.getElementById('domainFilter');
    const scopeFilter = document.getElementById('scopeFilter');
    const yearFilter = document.getElementById('yearFilter');
    document.getElementById('paperBrowserLink').href = payload.paper_browser_url || '#';
    document.getElementById('yearTrendsLink').href = payload.year_trends_url || '#';

    const domainRows = payload.domain_overview || [];
    const subdomainRows = payload.general_subdomain_overview || [];
    const yearRows = payload.year_overview || [];
    const domainYearRows = payload.domain_year_overview || {{}};
    const globalRows = payload.top_papers_global || [];
    const byDomainRows = payload.top_papers_by_domain || [];

    renderStats();
    renderOverviewTable(domainTable, ['Domain', 'Papers', 'Top Dataset', 'Top Metric', 'Top Concept'], domainRows.map(row => [row.domain, row.paper_count, row.top_dataset, row.top_metric, row.top_concept || '']));
    renderOverviewTable(subdomainTable, ['Subdomain', 'Papers', 'Top Dataset', 'Top Metric', 'Top Concept'], subdomainRows.map(row => [row.subdomain, row.paper_count, row.top_dataset, row.top_metric, row.top_concept || '']));
    renderOverviewTable(yearTable, ['Year', 'Total', 'SLAM', 'Robotics', 'General', 'SLAM-Supplement', 'Unknown'], yearRows.map(row => [row.year, row.paper_count, row.SLAM, row.Robotics, row.General, row['SLAM-Supplement'], row.Unknown]));
    renderOverviewTable(domainYearTable, ['Domain', 'Timeline'], Object.entries(domainYearRows).map(([domain, rows]) => [domain, rows.map(row => `${{row.year}}:${{row.paper_count}}`).join(' | ')]));
    populateDomainFilter();
    populateYearFilter();
    renderPaperTable();

    search.addEventListener('input', renderPaperTable);
    domainFilter.addEventListener('change', renderPaperTable);
    scopeFilter.addEventListener('change', renderPaperTable);
    yearFilter.addEventListener('change', renderPaperTable);

    function renderStats() {{
      const s = payload.summary || {{}};
      const cards = [
        ['Documents', s.document_count || 0],
        ['SLAM', (s.domain_counts || {{}}).SLAM || 0],
        ['Robotics', (s.domain_counts || {{}}).Robotics || 0],
        ['General', (s.domain_counts || {{}}).General || 0],
        ['Year Range', `${{((s.year_summary || {{}}).range || ['-', '-'])[0]}} to ${{((s.year_summary || {{}}).range || ['-', '-'])[1]}}`],
        ['Year Unknown', (s.year_summary || {{}}).unknown_count || 0],
        ['Resolved Titles', s.resolved_title_count || 0],
      ];
      stats.innerHTML = cards.map(([label, value]) => `
        <div class="stat">
          <div class="muted">${{label}}</div>
          <div class="value">${{value}}</div>
        </div>`).join('');
    }}

    function renderOverviewTable(target, headers, rows) {{
      target.innerHTML = `
        <thead>
          <tr>${{headers.map(header => `<th>${{header}}</th>`).join('')}}</tr>
        </thead>
        <tbody>
          ${{rows.map(row => `<tr>${{row.map(cell => `<td>${{escapeHtml(String(cell))}}</td>`).join('')}}</tr>`).join('')}}
        </tbody>`;
    }}

    function populateDomainFilter() {{
      const domains = ['all', ...new Set([...globalRows, ...byDomainRows].map(row => row.domain))];
      domainFilter.innerHTML = domains.map(domain => {{
        const label = domain === 'all' ? 'All domains' : domain;
        return `<option value="${{domain}}">${{label}}</option>`;
      }}).join('');
    }}

    function populateYearFilter() {{
      const years = ['all', ...new Set([...globalRows, ...byDomainRows].map(row => row.year).filter(Boolean))].sort((a, b) => a === 'all' ? -1 : Number(a) - Number(b));
      yearFilter.innerHTML = years.map(year => {{
        const label = year === 'all' ? 'All years' : year;
        return `<option value="${{year}}">${{label}}</option>`;
      }}).join('');
    }}

    function renderPaperTable() {{
      const term = search.value.trim().toLowerCase();
      const scope = scopeFilter.value;
      const domain = domainFilter.value;
      const year = yearFilter.value;
      const source = scope === 'global' ? globalRows : byDomainRows;
      const rows = source.filter(row => {{
        const matchDomain = domain === 'all' || row.domain === domain;
        const matchYear = year === 'all' || String(row.year) === year;
        const haystack = `${{row.label}} ${{row.source_file}} ${{row.original_label || ''}}`.toLowerCase();
        const matchTerm = !term || haystack.includes(term);
        return matchDomain && matchYear && matchTerm;
      }});

      paperTable.innerHTML = `
        <thead>
          <tr>
            <th>Rank</th>
            <th>Domain</th>
            <th>Year</th>
            <th>Title</th>
            <th>Degree</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          ${{rows.map(row => `
            <tr>
              <td>${{row.rank}}</td>
              <td>${{escapeHtml(row.domain)}}</td>
              <td>${{escapeHtml(String(row.year || ''))}}</td>
              <td>
                <div>${{row.paper_browser_row_url ? `<a href="${{row.paper_browser_row_url}}">${{escapeHtml(row.label)}}</a>` : escapeHtml(row.label)}}</div>
                ${{row.original_label ? `<div class="muted small">original: ${{escapeHtml(row.original_label)}}</div>` : ''}}
              </td>
              <td>${{row.degree}}</td>
              <td class="small">${{row.markdown_url ? `<a href="${{row.markdown_url}}">${{escapeHtml(row.source_file)}}</a>` : escapeHtml(row.source_file)}}</td>
            </tr>
          `).join('')}}
        </tbody>`;
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
