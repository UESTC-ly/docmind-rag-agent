"""Public dataset snapshot downloader safety and provenance tests."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from data import download_datasets


def _write_beir_zip(path, *, unsafe: bool = False) -> str:
    with zipfile.ZipFile(path, "w") as bundle:
        if unsafe:
            bundle.writestr("../escape.txt", "no")
        else:
            bundle.writestr("fixture/corpus.jsonl", '{"_id":"d1","text":"A"}\n')
            bundle.writestr("fixture/queries.jsonl", '{"_id":"q1","text":"Q"}\n')
            bundle.writestr(
                "fixture/qrels/test.tsv",
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
            )
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    _write_beir_zip(archive, unsafe=True)

    with pytest.raises(ValueError, match="unsafe archive member"):
        download_datasets._safe_extract_zip(archive, tmp_path / "out")


def test_download_beir_verifies_archive_and_records_file_hashes(
    tmp_path,
    monkeypatch,
):
    archive = tmp_path / "beir" / "fixture.zip"
    archive.parent.mkdir()
    expected_md5 = _write_beir_zip(archive)
    monkeypatch.setattr(download_datasets, "DATA_DIR", tmp_path)
    monkeypatch.setitem(
        download_datasets.BEIR_DATASETS,
        "fixture",
        {
            "download_uri": "https://example.test/fixture.zip",
            "md5": expected_md5,
            "source_name": "Fixture",
            "source_uri": "https://example.test/source",
            "source_version": "1",
            "license_name": "CC0",
            "split": "test",
        },
    )

    def reuse(_url, path, *, force):
        assert path == archive
        assert force is False
        return {
            "url": "https://example.test/fixture.zip",
            "path": "beir/fixture.zip",
            "bytes": archive.stat().st_size,
            "sha256": download_datasets._sha256(archive),
        }

    monkeypatch.setattr(download_datasets, "_download", reuse)

    report = download_datasets.download_beir("fixture", force=False)

    assert report["archive"]["md5"] == expected_md5
    assert report["license_name"] == "CC0"
    assert set(report["files"]) == {
        "corpus.jsonl",
        "queries.jsonl",
        "qrels/test.tsv",
    }
    assert all(len(value["sha256"]) == 64 for value in report["files"].values())


def test_existing_download_manifest_is_merged_safely(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"datasets": {"cmrc2018": {"source_version": "2018"}}}),
        encoding="utf-8",
    )

    assert download_datasets._load_existing_manifest_datasets(manifest) == {
        "cmrc2018": {"source_version": "2018"}
    }

    manifest.write_text("{broken", encoding="utf-8")
    assert download_datasets._load_existing_manifest_datasets(manifest) == {}
