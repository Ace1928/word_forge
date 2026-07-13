from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_KAIKKI_RAWDATA_URL = "https://kaikki.org/dictionary/rawdata.html"
_DEFAULT_BASE_URL = "https://kaikki.org/dictionary/"
_SIZE_UNITS = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4}
_LINK_PATTERN = re.compile(
    r'<a href="(?P<href>[^"]+/(?P<lang>[a-z0-9_-]+)-extract\.jsonl(?P<gzip>\.gz)?)">(?P<label>[^<]+)</a>',
    re.IGNORECASE,
)
_LABEL_PATTERN = re.compile(
    r"(?P<language_name>.+?)\s+(?P<lang>[a-z0-9_-]+)-extract\.jsonl(?:\.gz)?\s*\((?P<size_text>[^)]+)\)",
    re.IGNORECASE,
)


def _parse_size_bytes(size_text: str) -> int | None:
    text = str(size_text or "").strip().lower().replace("compressed", "").strip()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?b)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    return int(value * _SIZE_UNITS.get(unit, 1))


def _normalize_languages(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    raw: Iterable[str]
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = values
    seen: set[str] = set()
    normalized: list[str] = []
    for value in raw:
        item = str(value or "").strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def parse_kaikki_rawdata_html(
    html_text: str, *, page_url: str = DEFAULT_KAIKKI_RAWDATA_URL
) -> list[dict[str, Any]]:
    catalog_by_lang: dict[str, dict[str, Any]] = {}
    for match in _LINK_PATTERN.finditer(html_text):
        href = str(match.group("href") or "").strip()
        lang = str(match.group("lang") or "").strip().lower()
        label = str(match.group("label") or "").strip()
        gzip_suffix = str(match.group("gzip") or "")
        if not href or not lang:
            continue
        label_match = _LABEL_PATTERN.search(label)
        language_name = (
            str(label_match.group("language_name") or lang).strip() if label_match else lang
        )
        size_text = str(label_match.group("size_text") or "") if label_match else ""
        size_bytes = _parse_size_bytes(size_text)
        entry = catalog_by_lang.setdefault(
            lang,
            {
                "lang": lang,
                "language_name": language_name,
                "page_url": page_url,
                "jsonl_url": "",
                "jsonl_gz_url": "",
                "jsonl_bytes": None,
                "jsonl_gz_bytes": None,
            },
        )
        full_url = urllib.parse.urljoin(_DEFAULT_BASE_URL, href)
        if gzip_suffix:
            entry["jsonl_gz_url"] = full_url
            entry["jsonl_gz_bytes"] = size_bytes
        else:
            entry["jsonl_url"] = full_url
            entry["jsonl_bytes"] = size_bytes
    rows = [row for row in catalog_by_lang.values() if row.get("jsonl_gz_url")]
    rows.sort(key=lambda row: (row.get("jsonl_gz_bytes") or 10**18, row.get("lang") or ""))
    return rows


def read_kaikki_catalog(
    rawdata_html_path: Path, *, page_url: str = DEFAULT_KAIKKI_RAWDATA_URL
) -> list[dict[str, Any]]:
    return parse_kaikki_rawdata_html(
        rawdata_html_path.read_text(encoding="utf-8", errors="ignore"), page_url=page_url
    )


def build_kaikki_catalog_report(
    rawdata_html_path: Path,
    *,
    download_dir: Path,
    preferred_languages: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    preferred = _normalize_languages(preferred_languages)
    preferred_set = set(preferred)
    catalog = read_kaikki_catalog(rawdata_html_path)
    enriched: list[dict[str, Any]] = []
    for row in catalog:
        lang = str(row.get("lang") or "")
        gz_name = Path(urllib.parse.urlparse(str(row.get("jsonl_gz_url") or "")).path).name
        local_path = download_dir / gz_name
        enriched.append(
            {
                **row,
                "is_preferred": lang in preferred_set if preferred_set else False,
                "downloaded": local_path.exists(),
                "local_path": str(local_path),
                "local_size_bytes": int(local_path.stat().st_size) if local_path.exists() else 0,
            }
        )
    if preferred_set:
        enriched.sort(
            key=lambda row: (
                not row.get("is_preferred"),
                row.get("downloaded"),
                row.get("jsonl_gz_bytes") or 10**18,
                row.get("lang") or "",
            )
        )
    return {
        "contract": "eidos.word_forge.multilingual.sources.v1",
        "rawdata_html_path": str(rawdata_html_path),
        "download_dir": str(download_dir),
        "language_count": len(enriched),
        "downloaded_count": sum(1 for row in enriched if row.get("downloaded")),
        "preferred_languages": preferred,
        "preferred_count": sum(1 for row in enriched if row.get("is_preferred")),
        "entries": enriched,
    }


def choose_download_candidates(
    report: dict[str, Any],
    *,
    include_languages: Iterable[str] | str | None = None,
    max_languages: int | None = None,
    max_compressed_bytes: int | None = None,
    include_downloaded: bool = False,
) -> list[dict[str, Any]]:
    include = set(_normalize_languages(include_languages))
    rows = [row for row in report.get("entries", []) if isinstance(row, dict)]
    selected: list[dict[str, Any]] = []
    language_limit = (
        None if max_languages is None or int(max_languages) <= 0 else int(max_languages)
    )
    compressed_limit = (
        None
        if max_compressed_bytes is None or int(max_compressed_bytes) <= 0
        else int(max_compressed_bytes)
    )
    for row in rows:
        lang = str(row.get("lang") or "")
        if include and lang not in include:
            continue
        if not include_downloaded and row.get("downloaded"):
            continue
        size_bytes = row.get("jsonl_gz_bytes")
        if (
            compressed_limit is not None
            and isinstance(size_bytes, int)
            and size_bytes > compressed_limit
        ):
            continue
        selected.append(row)
        if language_limit is not None and len(selected) >= language_limit:
            break
    return selected


def download_catalog_entry(
    entry: dict[str, Any],
    *,
    download_dir: Path,
    force: bool = False,
    chunk_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    url = str(entry.get("jsonl_gz_url") or "").strip()
    if not url:
        raise ValueError("entry missing jsonl_gz_url")
    filename = (
        Path(urllib.parse.urlparse(url).path).name
        or f"{entry.get('lang') or 'unknown'}-extract.jsonl.gz"
    )
    target_path = download_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = entry.get("jsonl_gz_bytes")
    if target_path.exists() and not force:
        actual_size = int(target_path.stat().st_size)
        if not isinstance(expected_size, int) or actual_size == expected_size:
            return {
                "lang": entry.get("lang"),
                "status": "skipped",
                "path": str(target_path),
                "bytes": actual_size,
                "url": url,
            }
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as handle:
            while True:
                chunk = response.read(max(1024, int(chunk_bytes)))
                if not chunk:
                    break
                handle.write(chunk)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return {
        "lang": entry.get("lang"),
        "status": "downloaded",
        "path": str(target_path),
        "bytes": int(target_path.stat().st_size),
        "url": url,
    }
