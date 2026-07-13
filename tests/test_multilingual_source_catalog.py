from __future__ import annotations

from pathlib import Path

from word_forge.multilingual.runtime import (
    run_multilingual_batch_ingest,
    run_multilingual_source_catalog,
)
from word_forge.multilingual.source_catalog import (
    build_kaikki_catalog_report,
    choose_download_candidates,
)

SAMPLE_HTML = """
<ul>
<li><a href="downloads/pt/pt-extract.jsonl">Portuguese pt-extract.jsonl (332.0MB)</a></li>
<li><a href="downloads/pt/pt-extract.jsonl.gz">Portuguese pt-extract.jsonl.gz (compressed 33.9MB)</a></li>
<li><a href="downloads/it/it-extract.jsonl">Italian it-extract.jsonl (341.0MB)</a></li>
<li><a href="downloads/it/it-extract.jsonl.gz">Italian it-extract.jsonl.gz (compressed 35.0MB)</a></li>
<li><a href="downloads/fr/fr-extract.jsonl">French fr-extract.jsonl (6.1GB)</a></li>
<li><a href="downloads/fr/fr-extract.jsonl.gz">French fr-extract.jsonl.gz (compressed 662.7MB)</a></li>
</ul>
"""


def test_build_kaikki_catalog_report_parses_downloads(tmp_path: Path) -> None:
    rawdata_html_path = tmp_path / "kaikki-rawdata.html"
    rawdata_html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    (download_dir / "pt-extract.jsonl.gz").write_bytes(b"pt")

    report = build_kaikki_catalog_report(
        rawdata_html_path,
        download_dir=download_dir,
        preferred_languages=["pt", "it"],
    )

    assert report["contract"] == "eidos.word_forge.multilingual.sources.v1"
    assert report["language_count"] == 3
    assert report["downloaded_count"] == 1
    assert report["preferred_count"] == 2
    assert {report["entries"][0]["lang"], report["entries"][1]["lang"]} == {"pt", "it"}
    assert any(row["lang"] == "pt" and row["downloaded"] is True for row in report["entries"])

    candidates = choose_download_candidates(
        report, include_languages=["it", "fr"], max_compressed_bytes=40_000_000
    )
    assert [row["lang"] for row in candidates] == ["it"]


def test_choose_download_candidates_zero_caps_mean_unlimited(tmp_path: Path) -> None:
    rawdata_html_path = tmp_path / "kaikki-rawdata.html"
    rawdata_html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    report = build_kaikki_catalog_report(
        rawdata_html_path, download_dir=download_dir, preferred_languages=["pt", "it", "fr"]
    )
    candidates = choose_download_candidates(report, max_languages=0, max_compressed_bytes=0)
    assert [row["lang"] for row in candidates] == ["pt", "it", "fr"]


def test_run_multilingual_source_catalog_writes_report(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    rawdata_html_path = repo_root / "docs" / "research" / "kaikki-rawdata.html"
    rawdata_html_path.parent.mkdir(parents=True, exist_ok=True)
    rawdata_html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    download_dir = repo_root / "word_forge" / "data" / "multilingual" / "kaikki"

    result = run_multilingual_source_catalog(
        repo_root=repo_root,
        rawdata_html_path=rawdata_html_path,
        download_dir=download_dir,
        preferred_languages=["pt", "it"],
    )

    assert result["status"]["status"] == "completed"
    assert result["report"]["language_count"] == 3
    assert Path(result["artifacts"]["latest_json"]).exists()


def test_run_multilingual_batch_ingest_downloads_and_ingests_selected_languages(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    rawdata_html_path = repo_root / "docs" / "research" / "kaikki-rawdata.html"
    rawdata_html_path.parent.mkdir(parents=True, exist_ok=True)
    rawdata_html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    download_dir = repo_root / "word_forge" / "data" / "multilingual" / "kaikki"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    def _fake_download(entry, *, download_dir, force=False, chunk_bytes=1024 * 1024):
        lang = str(entry.get("lang") or "")
        target = download_dir / f"{lang}-extract.jsonl.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if lang == "pt":
            content = '{"word":"casa","lang":"Português","lang_code":"pt","pos":"noun","senses":[{"glosses":["house"]}],"translations":[{"lang":"English","lang_code":"en","word":"house"}]}\n'
        else:
            content = '{"word":"ciao","lang":"Italiano","lang_code":"it","pos":"interj","senses":[{"glosses":["hello"]}],"translations":[{"lang":"English","lang_code":"en","word":"hello"}]}\n'
        import gzip

        with gzip.open(target, "wt", encoding="utf-8") as handle:
            handle.write(content)
        return {
            "lang": lang,
            "status": "downloaded",
            "path": str(target),
            "bytes": int(target.stat().st_size),
            "url": str(entry.get("jsonl_gz_url") or ""),
        }

    monkeypatch.setattr("word_forge.multilingual.runtime.download_catalog_entry", _fake_download)

    result = run_multilingual_batch_ingest(
        repo_root=repo_root,
        rawdata_html_path=rawdata_html_path,
        download_dir=download_dir,
        db_path=db_path,
        include_languages=["pt", "it"],
        max_languages=2,
        force_download=True,
        force_ingest=True,
    )

    assert result["status"]["status"] == "completed"
    assert result["report"]["download"]["download_count"] == 2
    assert set(result["report"]["ingested_languages"]) == {"pt", "it"}
    assert Path(result["artifacts"]["latest_json"]).exists()


def test_run_multilingual_batch_ingest_uses_existing_downloads(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    rawdata_html_path = repo_root / "docs" / "research" / "kaikki-rawdata.html"
    rawdata_html_path.parent.mkdir(parents=True, exist_ok=True)
    rawdata_html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    download_dir = repo_root / "word_forge" / "data" / "multilingual" / "kaikki"
    download_dir.mkdir(parents=True, exist_ok=True)
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    import gzip

    with gzip.open(download_dir / "pt-extract.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            '{"word":"casa","lang":"Português","lang_code":"pt","pos":"noun","senses":[{"glosses":["house"]}],"translations":[{"lang":"English","lang_code":"en","word":"house"}]}\n'
        )

    result = run_multilingual_batch_ingest(
        repo_root=repo_root,
        rawdata_html_path=rawdata_html_path,
        download_dir=download_dir,
        db_path=db_path,
        include_languages=["pt"],
        max_languages=1,
        force_download=False,
        force_ingest=True,
    )

    assert result["status"]["status"] == "completed"
    assert result["report"]["download"]["results"][0]["status"] == "existing"
    assert result["report"]["ingested_languages"] == ["pt"]
