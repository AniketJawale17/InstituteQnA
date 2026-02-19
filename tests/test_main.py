from langchain_core.documents import Document

from institute_qna.data_preprocess.knoweldge_base_creation import KnowledgeBaseCreation


def _make_kb_stub() -> KnowledgeBaseCreation:
    """Create a lightweight instance without running heavy __init__."""
    kb = KnowledgeBaseCreation.__new__(KnowledgeBaseCreation)
    kb.university = "pune"
    kb._seen_content_hashes = set()
    return kb


def test_clean_markdown_content_removes_noise_patterns() -> None:
    kb = _make_kb_stub()
    raw = """
    Search Search
    Accessibility Tools

    Actual admission details are here.



    Contact us at admissions@example.edu
    """

    cleaned = kb.clean_markdown_content(raw)

    assert "Search Search" not in cleaned
    assert "Accessibility Tools" not in cleaned
    assert "Actual admission details are here." in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_metadata_from_content_finds_dates_email_phone() -> None:
    kb = _make_kb_stub()
    content = (
        "Admissions open from 12/03/2026. "
        "Email: ug-office@college.edu. "
        "Call 9876543210 for help."
    )

    metadata = kb.extract_metadata_from_content(content)

    assert "dates" in metadata and "12/03/2026" in metadata["dates"]
    assert "emails" in metadata and "ug-office@college.edu" in metadata["emails"]
    assert "phones" in metadata and "9876543210" in metadata["phones"]


def test_structure_documents_normalizes_page_and_adds_university() -> None:
    kb = _make_kb_stub()
    docs = [
        Document(
            page_content="Fee details",
            metadata={"source": "doc.pdf", "page": 4, "doc_type": "fees"},
        )
    ]

    structured = kb.structure_documents(docs)

    assert len(structured) == 1
    assert structured[0].metadata["source"] == "doc.pdf"
    assert structured[0].metadata["page"] == "4"
    assert structured[0].metadata["university"] == "pune"


def test_deduplicate_chunks_removes_exact_duplicates() -> None:
    kb = _make_kb_stub()
    docs = [
        Document(page_content="Admission process step by step", metadata={"source": "a"}),
        Document(page_content="Admission process step by step", metadata={"source": "b"}),
        Document(page_content="Hostel rules and regulations", metadata={"source": "c"}),
    ]

    deduped = kb.deduplicate_chunks(docs)

    assert len(deduped) == 2
    assert any(d.page_content == "Admission process step by step" for d in deduped)
    assert any(d.page_content == "Hostel rules and regulations" for d in deduped)


def test_classify_document_type_identifies_fees() -> None:
    kb = _make_kb_stub()
    doc_type = kb.classify_document_type(
        content="The tuition fees for first year are listed below.",
        source="https://example.edu/admissions",
        title="Fee Structure",
    )
    assert doc_type == "fees"
