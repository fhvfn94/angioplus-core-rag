# RAG Pipeline

# Этапы обработки данных

## 1. Ingestion
Документы:
- PDF
- DOCX

проходят через pipeline:

Вопрос
↓
Поиск отвтеа в Document
↓
Text extraction
↓
Chunking
↓
Embedding generation
↓
Qdrant storage