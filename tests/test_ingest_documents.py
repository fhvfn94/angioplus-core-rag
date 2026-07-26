import pytest

from scripts.ingest_documents import build_embedding_text


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (
            {
                "file_name": "AngioPlus Core v2.5 IFU_RU_Final.pdf",
                "document_type": "document",
                "section": "11 Инсталляция ПО",
                "page_start": 9,
                "page_end": 10,
                "text": "Установка программного обеспечения...",
            },
            "Документ: AngioPlus Core v2.5 IFU_RU_Final.pdf\nТип документа: document\nРаздел: 11 Инсталляция ПО\nСтраницы: 9-10\n\nУстановка программного обеспечения...",
        ),
        (
            {
                "file_name": "AngioPlus Core v2.5 IFU_RU_Final.pdf",
                "document_type": "document",
                "section": None,
                "page_start": 9,
                "page_end": 10,
                "text": "Текст chunk...",
            },
            "Документ: AngioPlus Core v2.5 IFU_RU_Final.pdf\nТип документа: document\nСтраницы: 9-10\n\nТекст chunk...",
        ),
        (
            {
                "file_name": "AngioPlus Core v2.5 IFU_RU_Final.pdf",
                "document_type": "document",
                "section": "11 Инсталляция ПО",
                "page_start": 9,
                "page_end": None,
                "text": "Текст chunk...",
            },
            "Документ: AngioPlus Core v2.5 IFU_RU_Final.pdf\nТип документа: document\nРаздел: 11 Инсталляция ПО\nСтраницы: 9\n\nТекст chunk...",
        ),
        (
            {
                "file_name": None,
                "document_type": None,
                "section": None,
                "page_start": None,
                "page_end": None,
                "text": "Текст без метаданных",
            },
            "Текст без метаданных",
        ),
        (
            {
                "file_name": "AngioPlus Core v2.5 IFU_RU_Final.pdf",
                "document_type": None,
                "section": "",
                "page_start": None,
                "page_end": None,
                "text": "Исходный текст chunk здесь",
            },
            "Документ: AngioPlus Core v2.5 IFU_RU_Final.pdf\n\nИсходный текст chunk здесь",
        ),
    ],
)
def test_build_embedding_text(metadata, expected):
    result = build_embedding_text(
        file_name=metadata["file_name"],
        document_type=metadata["document_type"],
        section=metadata["section"],
        page_start=metadata["page_start"],
        page_end=metadata["page_end"],
        text=metadata["text"],
    )
    assert result == expected


def test_build_embedding_text_does_not_stringify_none():
    result = build_embedding_text(
        file_name=None,
        document_type="document",
        section=None,
        page_start=None,
        page_end=None,
        text="Test text",
    )

    assert "None" not in result
    assert result == "Тип документа: document\n\nTest text"
