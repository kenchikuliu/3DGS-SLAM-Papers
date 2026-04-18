#!/usr/bin/env python3
"""
Resume-safe PDF to Markdown batch extractor.

Default behavior is local extraction via PyMuPDF because the remote MineRU API
was hitting file-size/page-count limits for this corpus. The script can still
attempt the API first when explicitly enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PDF_DIR = Path("C:/Users/Administrator/Downloads/3DGS-SLAM-Papers")
OUTPUT_DIR = PDF_DIR / "extracted_markdown"
STATE_FILE = PDF_DIR / "mineru_state.json"
LOG_FILE = PDF_DIR / "mineru_log.txt"

API_KEY = os.environ.get("MINERU_API_KEY", "")
API_BASE = "https://mineru.net/api/v1/agent"
PROXIES = {"https": "socks5h://127.0.0.1:3568", "http": "socks5h://127.0.0.1:3568"}
API_HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

POLL_SECS = 8
MAX_POLL = 150

_LOG = open(LOG_FILE, "a", encoding="utf-8", buffering=1)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _LOG.write(line + "\n")
    _LOG.flush()
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def save_state(completed: list[str], failed: list[str], pending: list[str]) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed": completed,
        "failed": failed,
        "pending": pending,
    }
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_local(pdf_path: Path, out_path: Path) -> bool:
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        pages_md = []
        for i, page in enumerate(doc):
            try:
                text = page.get_text("markdown")
            except Exception:
                text = page.get_text("text")
            if text and text.strip():
                pages_md.append(f"<!-- page {i + 1} -->\n{text.strip()}")
        doc.close()

        content = "\n\n".join(pages_md).strip()
        if not content:
            return False
        out_path.write_text(content + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        log(f"  local-error {pdf_path.stem[:60]}: {type(exc).__name__}: {exc}")
        return False


def extract_via_api(pdf_path: Path, out_path: Path) -> tuple[bool, str]:
    if not API_KEY:
        return False, "missing MINERU_API_KEY"

    try:
        create_resp = requests.post(
            f"{API_BASE}/parse/file",
            headers=API_HEADERS,
            json={
                "file_name": pdf_path.name,
                "language": "en",
                "is_ocr": False,
                "enable_formula": True,
            },
            proxies=PROXIES,
            timeout=30,
        )
        create_resp.raise_for_status()
        created = create_resp.json()
        if created.get("code") != 0:
            return False, created.get("msg", "unknown api error")

        task_id = created["data"]["task_id"]
        file_url = created["data"]["file_url"]
        with open(pdf_path, "rb") as fh:
            upload_resp = requests.put(file_url, data=fh, proxies=PROXIES, timeout=300)
        if upload_resp.status_code != 200:
            return False, f"upload HTTP {upload_resp.status_code}"

        for _ in range(MAX_POLL):
            time.sleep(POLL_SECS)
            poll_resp = requests.get(
                f"{API_BASE}/parse/{task_id}",
                headers=API_HEADERS,
                proxies=PROXIES,
                timeout=30,
            )
            poll_data = poll_resp.json().get("data", {})
            state = poll_data.get("state", "")
            if state == "done":
                md_url = poll_data.get("markdown_url")
                if not md_url:
                    return False, "no markdown_url"
                md_resp = requests.get(md_url, proxies=PROXIES, timeout=60)
                if md_resp.status_code != 200:
                    return False, f"download HTTP {md_resp.status_code}"
                out_path.write_text(md_resp.text, encoding="utf-8")
                return True, "api"
            if state == "failed":
                return False, poll_data.get("err_msg", "remote failed")
        return False, "poll timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def process_one(pdf_path: Path, local_only: bool) -> tuple[bool, str, str]:
    stem = pdf_path.stem
    out_path = OUTPUT_DIR / f"{stem}.md"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, stem, "already_exists"

    if not local_only:
        ok, reason = extract_via_api(pdf_path, out_path)
        if ok:
            log(f"  OK-API   {stem[:60]}")
            return True, stem, "api"
        log(f"  FALLBACK {stem[:60]} api_error={reason}")

    ok = extract_local(pdf_path, out_path)
    if ok:
        log(f"  OK-LOCAL {stem[:60]}")
        return True, stem, "local"

    return False, stem, "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-first", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    done_set = {p.stem for p in OUTPUT_DIR.glob("*.md") if p.stat().st_size > 0}
    pending = [p for p in all_pdfs if p.stem not in done_set]

    log("=== start ===")
    log(f"PDF total: {len(all_pdfs)}, existing md: {len(done_set)}, pending: {len(pending)}")

    completed = sorted(done_set)
    failed: list[str] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, pdf, not args.api_first): pdf for pdf in pending}
        for future in as_completed(futures):
            ok, stem, method = future.result()
            processed += 1
            if ok:
                if stem not in completed:
                    completed.append(stem)
            else:
                failed.append(stem)
                log(f"  FAIL     {stem[:60]} ({method})")

            if processed % 20 == 0 or processed == len(pending):
                remaining = [p.stem for p in pending if p.stem not in completed and p.stem not in failed]
                save_state(sorted(completed), sorted(failed), sorted(remaining))
                log(f"  progress {processed}/{len(pending)} ok={len(completed)} fail={len(failed)}")

    save_state(sorted(completed), sorted(failed), [])
    log(f"=== done ok={len(completed)} fail={len(failed)} ===")
    _LOG.close()


if __name__ == "__main__":
    main()
