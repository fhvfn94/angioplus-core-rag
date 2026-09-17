# -*- coding: utf-8 -*-
"""Tests for ``QdrantWriter``, ``IngestionPipeline`` and the ingestion CLI."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ingest_documents as ingest


class FakeQdrantClient:
    def __init__(self, url=None, api_key=None, collections=()):
        self.url = url
        self.api_key = api_key
        self._collections = list(collections)
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.upserts: list[dict] = []
        self.delete_error: Exception | None = None

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self._collections]
        )

    def create_collection(self, collection_name, vectors_config):
        self._collections.append(collection_name)
        self.created.append(
            {"name": collection_name, "size": vectors_config.size}
        )

    def delete_collection(self, collection_name):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(collection_name)
        if collection_name in self._collections:
            self._collections.remove(collection_name)

    def upsert(self, collection_name, points, wait):
        self.upserts.append(
            {"name": collection_name, "points": points, "wait": wait}
        )


@pytest.fixture()
def writer(monkeypatch):
    clients: list[FakeQdrantClient] = []

    def factory(url, api_key=None):
        client = FakeQdrantClient(url=url, api_key=api_key)
        clients.append(client)
        return client

    monkeypatch.setattr(ingest, "QdrantClient", factory)
    return ingest.QdrantWriter(
        url="http://localhost:6333",
        collection_name="col",
        api_key="secret",
    )


def test_writer_passes_connection_settings(writer):
    assert writer.client.url == "http://localhost:6333"
    assert writer.client.api_key == "secret"


def test_ensure_collection_creates_missing_collection(writer):
    writer.ensure_collection(vector_size=4)

    assert writer.client.created == [{"name": "col", "size": 4}]


def test_ensure_collection_is_noop_when_present(writer):
    writer.client._collections.append("col")

    writer.ensure_collection(vector_size=4)

    assert writer.client.created == []


def test_recreate_collection_deletes_then_creates(writer):
    writer.client._collections.append("col")

    writer.recreate_collection(vector_size=3)

    assert writer.client.deleted == ["col"]
    assert writer.client.created == [{"name": "col", "size": 3}]


def test_delete_collection_swallows_errors(writer):
    writer.client.delete_error = RuntimeError("does not exist")

    writer.delete_collection_if_exists()

    assert writer.client.deleted == []


def test_upsert_is_noop_without_chunks(writer):
    writer.upsert(payload_chunks=[], vectors=[], vector_size=3)

    assert writer.client.upserts == []


def test_upsert_writes_points_with_payload(writer):
    chunk = ingest.Chunk(
        id="11111111-1111-1111-1111-111111111111",
        text="текст",
        metadata={"file_name": "IFU.pdf", "page_start": 1},
    )

    writer.upsert(payload_chunks=[chunk], vectors=[[0.1, 0.2]], vector_size=2)

    call = writer.client.upserts[0]
    assert call["name"] == "col"
    assert call["wait"] is True
    point = call["points"][0]
    assert point.id == chunk.id
    assert point.payload == {
        "text": "текст",
        "file_name": "IFU.pdf",
        "page_start": 1,
    }
    assert writer.client.created == [{"name": "col", "size": 2}]


class StubExtractor(ingest.SegmentExtractor):
    def __init__(self, segments):
        self.segments = segments

    def extract_segments(self, path: Path):
        return list(self.segments)


class StubExtractorFactory(ingest.ExtractorFactory):
    def __init__(self, segments_by_suffix: dict[str, list]):
        self.segments_by_suffix = segments_by_suffix

    def get(self, path: Path):
        return StubExtractor(self.segments_by_suffix[path.suffix.lower()])


class RecordingWriter:
    def __init__(self):
        self.ensured: list[int] = []
        self.recreated: list[int] = []
        self.upserts: list[dict] = []

    def ensure_collection(self, vector_size: int) -> None:
        self.ensured.append(vector_size)

    def recreate_collection(self, vector_size: int) -> None:
        self.recreated.append(vector_size)

    def upsert(self, payload_chunks, vectors, *, vector_size: int) -> None:
        self.upserts.append(
            {
                "chunks": list(payload_chunks),
                "vectors": list(vectors),
                "vector_size": vector_size,
            }
        )


def make_pipeline(
    segments_by_suffix: dict[str, list],
    *,
    recreate_collection: bool = False,
    writer: RecordingWriter | None = None,
) -> tuple[ingest.IngestionPipeline, RecordingWriter]:
    writer = writer or RecordingWriter()
    pipeline = ingest.IngestionPipeline(
        extractor_factory=StubExtractorFactory(segments_by_suffix),
        classifier=ingest.DocumentClassifier(),
        chunker=ingest.SemanticChunker(min_tokens=1, max_tokens=1000),
        embedder=ingest.HashEmbedder(dimension=8),
        writer=writer,
        recreate_collection=recreate_collection,
    )
    return pipeline, writer


@pytest.mark.parametrize(
    "section, chunk_text, expected",
    [
        ("11 Инсталляция", "Любой текст", "11 Инсталляция"),
        ("General", "Первое предложение. Второе.", "Первое предложение"),
        ("", "", "General"),
    ],
)
def test_infer_topic(section, chunk_text, expected):
    assert ingest.IngestionPipeline._infer_topic(section, chunk_text) == expected


def test_infer_topic_truncates_long_sections():
    long_section = "9 " + "x" * 400

    topic = ingest.IngestionPipeline._infer_topic(long_section, "text")

    assert len(topic) == 200


def test_collect_documents_filters_and_sorts(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.PDF").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "nested" / "c.docx").touch()

    recursive = ingest.IngestionPipeline._collect_documents(tmp_path, recursive=True)
    flat = ingest.IngestionPipeline._collect_documents(tmp_path, recursive=False)

    assert [path.name for path in recursive] == ["a.PDF", "b.pdf", "c.docx"]
    assert [path.name for path in flat] == ["a.PDF", "b.pdf"]


def test_collect_documents_requires_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        ingest.IngestionPipeline._collect_documents(tmp_path / "missing", recursive=True)


def test_run_returns_zero_counts_without_documents(tmp_path):
    pipeline, writer = make_pipeline({})

    assert pipeline.run(tmp_path) == {"documents": 0, "chunks": 0}
    assert writer.upserts == []


def test_run_skips_documents_without_segments(tmp_path):
    (tmp_path / "empty.pdf").touch()
    pipeline, writer = make_pipeline({".pdf": []})

    assert pipeline.run(tmp_path) == {"documents": 1, "chunks": 0}
    assert writer.upserts == []


def test_run_ingests_pdf_chunks(tmp_path):
    (tmp_path / "AngioPlus Core v2.5 IFU_RU.pdf").touch()
    pipeline, writer = make_pipeline(
        {".pdf": [(9, "11 Инсталляция ПО\nУстановите ПО. Внимание: риск.")]}
    )

    result = pipeline.run(tmp_path, recursive=False)

    assert result == {"documents": 1, "chunks": 1}
    assert writer.ensured == [8]
    assert writer.recreated == []
    chunk = writer.upserts[0]["chunks"][0]
    assert chunk.metadata["source"] == "IFU"
    assert chunk.metadata["support_level"] == "L2"
    assert chunk.metadata["risk_level"] == "warning"
    assert chunk.metadata["version"] == "2.5"
    assert chunk.metadata["section"] == "11 Инсталляция ПО"
    assert chunk.metadata["document_type"] == "document"
    assert chunk.metadata["language"] == "ru"
    assert chunk.metadata["page_start"] == 9
    assert len(writer.upserts[0]["vectors"][0]) == 8


def test_run_recreates_collection_when_requested(tmp_path):
    (tmp_path / "guide.pdf").touch()
    pipeline, writer = make_pipeline(
        {".pdf": [(1, "Текст документа")]},
        recreate_collection=True,
    )

    pipeline.run(tmp_path)

    assert writer.recreated == [8]
    assert writer.ensured == []


def test_run_builds_one_chunk_per_faq_row(tmp_path):
    (tmp_path / "FAQ.xlsx").touch()
    row_text = (
        "Source Type / Тип источника: FAQ\n"
        "Sheet / Лист: Flow Velocity\n"
        "Row / Строка: 5\n\n"
        "Question / Вопрос:\nWhat is FFR?\n\n"
        "Answer / Ответ:\nA ratio."
    )
    pipeline, writer = make_pipeline({".xlsx": [(5, row_text), (6, "Без метаданных")]})

    result = pipeline.run(tmp_path)

    assert result == {"documents": 1, "chunks": 2}
    first, second = writer.upserts[0]["chunks"]
    assert first.metadata["source"] == "FAQ"
    assert first.metadata["document_type"] == "faq"
    assert first.metadata["sheet_name"] == "Flow Velocity"
    assert first.metadata["topic"] == "What is FFR?"
    assert first.metadata["row_number"] == 5
    assert first.metadata["page_start"] == first.metadata["page_end"] == 5
    assert second.metadata["sheet_name"] == "Unknown"
    assert second.metadata["topic"] == "Unknown"


def test_run_reports_when_no_chunks_are_generated(tmp_path):
    (tmp_path / "toc.pdf").touch()
    toc_segments = [(1, "\n\n".join(f"Раздел {i} ....... {i}" for i in range(1, 7)))]
    pipeline, writer = make_pipeline({".pdf": toc_segments})

    assert pipeline.run(tmp_path) == {"documents": 1, "chunks": 0}
    assert writer.upserts == []


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(ingest.sys, "argv", ["ingest_documents.py"])
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    args = ingest.parse_args()

    assert args.input_dir == Path("data/training_materials")
    assert args.collection == ingest.DEFAULT_COLLECTION
    assert args.recursive is True
    assert args.recreate_collection is False
    assert args.debug_hash_embeddings is False


def test_parse_args_overrides(monkeypatch):
    monkeypatch.setattr(
        ingest.sys,
        "argv",
        [
            "ingest_documents.py",
            "--input-dir",
            "docs",
            "--collection",
            "other",
            "--recreate-collection",
            "--debug-hash-embeddings",
            "--no-recursive",
        ],
    )

    args = ingest.parse_args()

    assert args.input_dir == Path("docs")
    assert args.collection == "other"
    assert args.recreate_collection is True
    assert args.debug_hash_embeddings is True
    assert args.recursive is False


def test_main_wires_pipeline_from_arguments(monkeypatch):
    monkeypatch.setattr(
        ingest.sys,
        "argv",
        [
            "ingest_documents.py",
            "--input-dir",
            "docs",
            "--debug-hash-embeddings",
            "--collection",
            "other",
        ],
    )
    captured: dict = {}

    class FakeWriter:
        def __init__(self, url, collection_name, api_key=None):
            captured["writer"] = {
                "url": url,
                "collection": collection_name,
                "api_key": api_key,
            }

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def run(self, input_dir, recursive):
            captured["run"] = {"input_dir": input_dir, "recursive": recursive}
            return {"documents": 1, "chunks": 2}

    monkeypatch.setattr(ingest, "QdrantWriter", FakeWriter)
    monkeypatch.setattr(ingest, "IngestionPipeline", FakePipeline)

    ingest.main()

    assert captured["writer"]["collection"] == "other"
    assert captured["run"] == {"input_dir": Path("docs"), "recursive": True}
    assert isinstance(captured["pipeline"]["embedder"], ingest.HashEmbedder)
    assert captured["pipeline"]["recreate_collection"] is False
