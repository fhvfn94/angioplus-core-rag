# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}
DEFAULT_INDEX_NAME = "angioplus-kb-v2"


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict


def build_embedding_text(
    file_name: str | None,
    document_type: str | None,
    section: str | None,
    page_start: int | None,
    page_end: int | None,
    text: str,
) -> str:
    parts: list[str] = []

    if file_name:
        parts.append(f"Документ: {file_name}")
    if document_type:
        parts.append(f"Тип документа: {document_type}")
    if section:
        parts.append(f"Раздел: {section}")

    if page_start is not None or page_end is not None:
        if page_start is not None and page_end is not None:
            parts.append(f"Страницы: {page_start}-{page_end}")
        elif page_start is not None:
            parts.append(f"Страницы: {page_start}")
        else:
            parts.append(f"Страницы: {page_end}")

    if parts:
        parts.append("")
    parts.append(text or "")
    return "\n".join(parts)


# ============================================================
# ИЗВЛЕЧЕНИЕ ТЕКСТА (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

class SegmentExtractor:
    def extract_segments(self, path: Path) -> list[tuple[int, str]]:
        raise NotImplementedError


class PdfSegmentExtractor(SegmentExtractor):
    def extract_segments(self, path: Path) -> list[tuple[int, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pypdf") from exc

        reader = PdfReader(str(path))
        segments: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            normalized = raw.replace("\r", "\n")
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
            blocks = [b.strip() for b in normalized.split("\n\n") if b.strip()]
            for block in blocks:
                segments.append((idx, block))
        return segments


class DocxSegmentExtractor(SegmentExtractor):
    def extract_segments(self, path: Path) -> list[tuple[int, str]]:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Missing dependency: python-docx") from exc

        doc = Document(str(path))
        segments: list[tuple[int, str]] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                segments.append((1, text))
        return segments


class ExcelFaqSegmentExtractor(SegmentExtractor):
    SKIP_SHEETS = {"WpsReserved_CellImgList"}

    def extract_segments(self, path: Path) -> list[tuple[int, str]]:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise RuntimeError("Missing dependency: openpyxl") from exc

        workbook = load_workbook(filename=str(path), data_only=True)
        segments: list[tuple[int, str]] = []

        for sheet_name in workbook.sheetnames:
            if sheet_name in self.SKIP_SHEETS:
                continue

            sheet = workbook[sheet_name]
            rows = list(sheet.iter_rows(values_only=True))

            if not rows:
                continue

            headers = [self._clean_cell(value) for value in rows[0]]
            headers = [
                header if header else f"Column {idx + 1}"
                for idx, header in enumerate(headers)
            ]

            for row_index, row in enumerate(rows[1:], start=2):
                values = [self._clean_cell(value) for value in row]

                if not any(values):
                    continue

                values_cf = [value.casefold() for value in values if value]

                if "question" in values_cf and "answer" in values_cf:
                    continue

                non_empty_values = [value for value in values if value]
                if len(non_empty_values) == 1:
                    continue

                question_candidate = values[1] if len(values) > 1 else ""
                answer_candidate = values[2] if len(values) > 2 else ""

                if not question_candidate or not answer_candidate:
                    continue

                if question_candidate.casefold() == "question" or answer_candidate.casefold() == "answer":
                    continue

                row_text_parts = [
                    f"Source Type / Тип источника: FAQ",
                    f"Language / Язык: English",
                    f"Sheet / Лист: {sheet_name}",
                    f"Row / Строка: {row_index}",
                    "",
                ]

                for header, value in zip(headers, values):
                    if not value:
                        continue

                    normalized_header = self._normalize_header(header)

                    if normalized_header in {"question", "q"}:
                        row_text_parts.append(f"Question / Вопрос:\n{value}")
                    elif normalized_header in {"answer", "a"}:
                        row_text_parts.append(f"Answer / Ответ:\n{value}")
                    elif normalized_header in {"category", "topic"}:
                        row_text_parts.append(f"Category / Категория:\n{value}")
                    else:
                        row_text_parts.append(f"{header}:\n{value}")

                    row_text_parts.append("")

                text = "\n".join(row_text_parts).strip()
                if text:
                    segments.append((row_index, text))

        return segments

    @staticmethod
    def _clean_cell(value) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_header(header: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", header.casefold())


class ExtractorFactory:
    def __init__(self) -> None:
        self._extractors: dict[str, SegmentExtractor] = {
            ".pdf": PdfSegmentExtractor(),
            ".docx": DocxSegmentExtractor(),
            ".xlsx": ExcelFaqSegmentExtractor(),
        }

    def get(self, path: Path) -> SegmentExtractor:
        ext = path.suffix.lower()
        extractor = self._extractors.get(ext)
        if extractor is None:
            raise ValueError(f"Unsupported extension: {ext}")
        return extractor


# ============================================================
# КЛАССИФИКАЦИЯ (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

class DocumentClassifier:
    def classify_source(self, path: Path) -> str:
        name_lower = path.name.lower()
        stem_lower = path.stem.lower()
        if path.suffix.lower() == ".xlsx":
            return "FAQ"

        if "angioplus core" in name_lower and "ifu" in name_lower:
            return "IFU"
        if stem_lower == "2025july" or name_lower.startswith("2025july"):
            return "TRAINING"
        if "competitor" in name_lower:
            return "COMMERCIAL"
        if "commercial" in name_lower or "distributor" in name_lower:
            return "COMMERCIAL"

        if ("l1" in stem_lower and "l2" in stem_lower) or "l1_l2" in stem_lower:
            return "L1L2"
        full = path.name.casefold()
        if "l1" in full and "l2" in full:
            return "L1L2"

        if "ifu" in name_lower:
            return "IFU"
        if "training" in name_lower:
            return "TRAINING"
        return "COMMERCIAL"

    def infer_support_level(self, source: str) -> str:
        if source == "IFU":
            return "L2"
        if source == "FAQ":
            return "L2"
        if source == "TRAINING":
            return "L1"
        if source == "L1L2":
            return "L2"
        return "L1"

    def infer_risk_level(self, chunk_text: str) -> str:
        text_cf = chunk_text.casefold()

        critical_terms_ru = (
            "\u0441\u0435\u0440\u044c\u0435\u0437\u043d\u044b\u0439 \u0438\u043d\u0446\u0438\u0434\u0435\u043d\u0442",
            "\u0441\u0435\u0440\u044c\u0451\u0437\u043d\u044b\u0439 \u0438\u043d\u0446\u0438\u0434\u0435\u043d\u0442",
            "\u0430\u0432\u0430\u0440\u0438\u044f",
            "\u043a\u0440\u0438\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0439",
            "\u0441\u0431\u043e\u0439",
            "\u043e\u0442\u043a\u0430\u0437",
            "\u043d\u0435\u0438\u0441\u043f\u0440\u0430\u0432\u043d\u043e\u0441\u0442\u044c",
            "\u043f\u043e\u0442\u0435\u0440\u044f \u0434\u0430\u043d\u043d\u044b\u0445",
        )
        critical_terms_en = ("failure", "critical", "hazard", "emergency", "fatal")

        warning_terms_ru = (
            "\u043e\u0448\u0438\u0431\u043a\u0430",
            "\u0440\u0438\u0441\u043a",
            "\u043f\u0440\u0435\u0434\u0443\u043f\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u0435",
            "\u0432\u043d\u0438\u043c\u0430\u043d\u0438\u0435",
        )
        warning_terms_en = ("warning", "caution")

        def _hits(terms: tuple[str, ...]) -> bool:
            return any(term.casefold() in text_cf for term in terms)

        if _hits(critical_terms_ru) or _hits(critical_terms_en):
            return "critical"
        if _hits(warning_terms_ru) or _hits(warning_terms_en):
            return "warning"
        return "normal"

    def infer_version(self, path: Path) -> str:
        match = re.search(r"v(\d+(?:\.\d+)*)", path.stem.lower())
        if match:
            return match.group(1)
        return "unknown"


# ============================================================
# ЧАНКИНГ (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

def _is_heading_line(line_raw: str) -> bool:
    line = line_raw.strip()
    if "..." in line or "…" in line:
        return False
    if re.search(r"\.{5,}", line):
        return False
    if not line or "\n" in line:
        return False
    if len(line) > 220:
        return False
    m = re.match(r"^(\d{1,3})\s+(.+)$", line)
    if not m:
        return False
    title = m.group(2).strip()
    if not re.match(r"^[\w\u0400-\u04FF]", title):
        return False
    if len(title) < 3:
        return False
    return True


def _segments_to_sectioned_segments(segments: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    section = "General"
    tagged: list[tuple[int, str, str]] = []

    def flush(lines: list[str], page_no: int) -> None:
        text = "\n".join(lines).strip()
        if text:
            tagged.append((page_no, text, section))

    for page_no, raw_block in segments:
        block = raw_block.replace("\r", "\n")
        lines_queue = [ln.rstrip() for ln in block.split("\n")]
        paragraph_buf: list[str] = []

        def flush_para() -> None:
            nonlocal paragraph_buf
            flush(paragraph_buf, page_no)
            paragraph_buf = []

        idx = 0
        while idx < len(lines_queue):
            ln = lines_queue[idx].strip()
            if not ln:
                flush_para()
                idx += 1
                continue
            if _is_heading_line(ln):
                flush_para()
                section = ln
                idx += 1
                continue
            paragraph_buf.append(ln.strip())
            idx += 1
        flush_para()

    return tagged


class SemanticChunker:
    def __init__(self, min_tokens: int = 120, max_tokens: int = 500, overlap_tokens: int = 50) -> None:
        if min_tokens <= 0 or max_tokens <= 0 or min_tokens > max_tokens:
            raise ValueError("Invalid chunk token limits")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        words = len(text.split())
        return max(1, int(words / 0.75))

    @staticmethod
    def _is_toc_like(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 5:
            return False
        dotted_lines = sum(1 for line in lines if "..." in line or "…" in line)
        return dotted_lines >= 3

    def split(self, segments: list[tuple[int, str]]) -> list[tuple[str, str, int, int]]:
        body_blocks = _segments_to_sectioned_segments(segments)
        if not body_blocks:
            return []

        chunks: list[tuple[str, str, int, int]] = []
        current_blocks: list[tuple[int, str, str]] = []
        current_tokens = 0
        current_section: str | None = None

        def make_chunk(blocks: list[tuple[int, str, str]]) -> tuple[str, str, int, int] | None:
            if not blocks:
                return None
            pages = [page for page, _, _ in blocks]
            section = blocks[0][2]
            text = "\n\n".join(block_text for _, block_text, _ in blocks).strip()
            if not text:
                return None
            if self._is_toc_like(text):
                return None
            return section, text, min(pages), max(pages)

        def flush_chunk() -> None:
            nonlocal current_blocks, current_tokens
            chunk = make_chunk(current_blocks)
            if chunk is not None:
                chunks.append(chunk)
            current_blocks = []
            current_tokens = 0

        for page_no, text, section in body_blocks:
            block_tokens = self._estimate_tokens(text)

            if current_section is None:
                current_section = section

            section_changed = section != current_section

            if section_changed:
                flush_chunk()
                current_section = section

            if current_blocks and current_tokens + block_tokens > self.max_tokens:
                flush_chunk()

            current_blocks.append((page_no, text, section))
            current_tokens += block_tokens

            if current_tokens >= self.max_tokens:
                flush_chunk()

        if current_blocks:
            chunk = make_chunk(current_blocks)
            if chunk is not None:
                if self._estimate_tokens(chunk[1]) < self.min_tokens and chunks and chunks[-1][0] == chunk[0]:
                    prev_section, prev_text, prev_start, prev_end = chunks[-1]
                    chunks[-1] = (
                        prev_section,
                        f"{prev_text}\n\n{chunk[1]}",
                        min(prev_start, chunk[2]),
                        max(prev_end, chunk[3]),
                    )
                else:
                    chunks.append(chunk)

        return chunks


# ============================================================
# YANDEX VECTOR STORE WRITER (ЗАМЕНА QdrantWriter)
# ============================================================

class YandexVectorStoreWriter:
    """Загружает чанки в Yandex Vector Store через OpenAI-совместимый SDK."""

    def __init__(self, api_key: str, folder_id: str, index_name: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            project=folder_id,
        )
        self.index_name = index_name

    def upload_chunks(self, chunks: list[Chunk]) -> str:
        # 1. Записываем чанки во временный JSONL
        jsonl_path = Path("_temp_chunks.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                body = build_embedding_text(
                    file_name=chunk.metadata.get("file_name"),
                    document_type=chunk.metadata.get("document_type"),
                    section=chunk.metadata.get("section"),
                    page_start=chunk.metadata.get("page_start"),
                    page_end=chunk.metadata.get("page_end"),
                    text=chunk.text,
                )
                if len(body) > 8000:
                    body = body[:7997] + "..."
                f.write(json.dumps({"body": body}, ensure_ascii=False) + "\n")

        print(f"📝 JSONL записан: {len(chunks)} чанков → {jsonl_path}")

        # 2. Загружаем JSONL файл (OpenAI SDK — как в доках Яндекса)
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(
                file=(jsonl_path.name, f, "application/jsonlines"),
                purpose="assistants",
                extra_body={"format": "chunks"},
            )
        print(f"📁 Файл загружен: {file_obj.id}")

        # 3. Создаём Vector Store
        vs = self.client.vector_stores.create(
            name=self.index_name,
        )
        print(f"✅ Vector Store создан: {vs.id}")

        # 4. Привязываем файл с format=chunks
        #    chunking_strategy и attributes передаём через extra_body,
        #    т.к. это расширения Яндекса поверх OpenAI API
        self.client.vector_stores.files.create(
            vector_store_id=vs.id,
            file_id=file_obj.id,
            extra_body={
                "chunking_strategy": {
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 1408,
                        "chunk_overlap_tokens": 148,
                    },
                },
                "attributes": {"format": "chunks"},
            },
        )
        print(f"📎 Файл привязан (format=chunks)")

        # 5. Ждём завершения индексации
        print("⏳ Ожидание индексации...")
        while True:
            vs_file = self.client.vector_stores.files.retrieve(
                vector_store_id=vs.id,
                file_id=file_obj.id,
            )
            status = vs_file.status
            print(f"   Статус: {status}")
            if status == "completed":
                break
            elif status == "failed":
                raise RuntimeError(f"Индексация не удалась: {vs_file.last_error}")
            time.sleep(5)

        # 6. Удаляем временный файл
        jsonl_path.unlink()
        print(f"✅ Готово! Чанков: {len(chunks)}, Store ID: {vs.id}")
        return vs.id


# ============================================================
# PIPELINE (ОБНОВЛЁН: убраны эмбеддинги, Qdrant)
# ============================================================

class IngestionPipeline:
    def __init__(
        self,
        extractor_factory: ExtractorFactory,
        classifier: DocumentClassifier,
        chunker: SemanticChunker,
        writer: YandexVectorStoreWriter,
    ) -> None:
        self.extractor_factory = extractor_factory
        self.classifier = classifier
        self.chunker = chunker
        self.writer = writer

    def _load_segments(self, path: Path) -> tuple[list[tuple[int, str]], str, str]:
        extractor = self.extractor_factory.get(path)
        segments = extractor.extract_segments(path)
        source = self.classifier.classify_source(path)
        version = self.classifier.infer_version(path)
        return segments, source, version

    @staticmethod
    def _infer_topic(section: str, chunk_text: str) -> str:
        if section and section != "General":
            return section[:200]
        first_sentence = re.split(r"[.!?]\s+", chunk_text.strip())[0]
        return first_sentence[:120] if first_sentence else "General"

    def _build_chunks(
        self,
        path: Path,
        segments: list[tuple[int, str]],
        source: str,
        version: str,
    ) -> list[Chunk]:
        support_level = self.classifier.infer_support_level(source)
        output: list[Chunk] = []

        if path.suffix.lower() == ".xlsx":
            for row_number, row_text in segments:
                sheet_match = re.search(r"Sheet / Лист:\s*(.+)", row_text)
                question_match = re.search(
                    r"Question / Вопрос:\s*(.+?)(?:\n\n|Answer / Ответ:|$)",
                    row_text,
                    flags=re.DOTALL,
                )

                sheet_name = sheet_match.group(1).strip() if sheet_match else "Unknown"
                question = question_match.group(1).strip() if question_match else ""

                metadata = {
                    "source": "FAQ",
                    "section": sheet_name,
                    "topic": question[:200] if question else sheet_name,
                    "support_level": support_level,
                    "risk_level": self.classifier.infer_risk_level(row_text),
                    "version": version,
                    "page_start": row_number,
                    "page_end": row_number,
                    "row_number": row_number,
                    "sheet_name": sheet_name,
                    "file_name": path.name,
                    "file_path": str(path.resolve()),
                    "language": "en",
                    "document_type": "faq",
                }

                output.append(Chunk(id=str(uuid.uuid4()), text=row_text, metadata=metadata))
            return output

        chunk_items = self.chunker.split(segments)

        for section, chunk_text, page_start, page_end in chunk_items:
            topic = self._infer_topic(section, chunk_text)
            metadata = {
                "source": source,
                "section": section,
                "topic": topic,
                "support_level": support_level,
                "risk_level": self.classifier.infer_risk_level(chunk_text),
                "version": version,
                "page_start": page_start,
                "page_end": page_end,
                "file_name": path.name,
                "file_path": str(path.resolve()),
                "language": "ru",
                "document_type": "document",
            }
            output.append(Chunk(id=str(uuid.uuid4()), text=chunk_text, metadata=metadata))

        return output

    def run(self, input_dir: Path, recursive: bool = True) -> dict[str, int]:
        paths = self._collect_documents(input_dir=input_dir, recursive=recursive)
        if not paths:
            logging.warning("No supported documents found in %s", input_dir)
            return {"documents": 0, "chunks": 0}

        all_chunks: list[Chunk] = []
        doc_profiles: dict[str, dict[str, int | str]] = {}

        for path in paths:
            segments, source, version = self._load_segments(path)
            char_count = sum(len(t) for _, t in segments)

            if not segments:
                logging.warning("Skipping empty document: %s", path)
                continue

            chunks = self._build_chunks(path, segments, source, version)
            all_chunks.extend(chunks)
            key = str(path.resolve())
            doc_profiles[key] = {
                "name": path.name,
                "source": source,
                "chars": char_count,
                "chunks": len(chunks),
            }

        if not all_chunks:
            logging.warning("No chunks generated.")
            return {"documents": len(paths), "chunks": 0}

        # DEBUG: сохраняем сформированные чанки для проверки retrieval
        debug_path = Path("_debug_chunks.jsonl")

        with open(debug_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(
                    json.dumps(
                        {
                            "file_name": chunk.metadata.get("file_name"),
                            "section": chunk.metadata.get("section"),
                            "page_start": chunk.metadata.get("page_start"),
                            "page_end": chunk.metadata.get("page_end"),
                            "text": chunk.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f"🔎 DEBUG чанки сохранены → {debug_path}")



        # Логирование статистики
        for fp, stats in sorted(doc_profiles.items(), key=lambda x: str(x[1]["name"])):
            logging.info(
                "Document=%r source=%s chars=%s chunks=%s",
                stats["name"],
                stats["source"],
                stats["chars"],
                stats["chunks"],
            )

        # Загружаем в Yandex Vector Store (вместо Qdrant + эмбеддингов)
        store_id = self.writer.upload_chunks(all_chunks)

        return {"documents": len(paths), "chunks": len(all_chunks), "store_id": store_id}

    @staticmethod
    def _collect_documents(input_dir: Path, recursive: bool) -> list[Path]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        iterator: Iterable[Path]
        if recursive:
            iterator = input_dir.rglob("*")
        else:
            iterator = input_dir.glob("*")
        return sorted(
            [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        )


# ============================================================
# CLI (ОБНОВЛЁН: убраны Qdrant/Gemini флаги)
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDF/DOCX/XLSX into Yandex Vector Store for AngioPlus RAG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/training_materials"),
        help="Path to documents directory.",
    )
    parser.add_argument(
        "--yandex-api-key",
        default=os.getenv("YANDEX_API_KEY"),
        help="Yandex AI Studio API key.",
    )
    parser.add_argument(
        "--yandex-folder-id",
        default=os.getenv("YANDEX_FOLDER_ID"),
        help="Yandex Cloud folder ID.",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Vector Store name.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan input-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    api_key = args.yandex_api_key
    folder_id = args.yandex_folder_id

    if not api_key:
        sys.exit("❌ YANDEX_API_KEY не задан. Укажите через --yandex-api-key или в .env")
    if not folder_id:
        sys.exit("❌ YANDEX_FOLDER_ID не задан. Укажите через --yandex-folder-id или в .env")

    writer = YandexVectorStoreWriter(
        api_key=api_key,
        folder_id=folder_id,
        index_name=args.index_name,
    )

    pipeline = IngestionPipeline(
        extractor_factory=ExtractorFactory(),
        classifier=DocumentClassifier(),
        chunker=SemanticChunker(min_tokens=120, max_tokens=500, overlap_tokens=50),
        writer=writer,
    )

    result = pipeline.run(input_dir=args.input_dir, recursive=args.recursive)
    logging.info(
        "Ingestion done. Documents: %s, chunks: %s, store_id: %s",
        result["documents"],
        result["chunks"],
        result.get("store_id", "N/A"),
    )


if __name__ == "__main__":
    main()
