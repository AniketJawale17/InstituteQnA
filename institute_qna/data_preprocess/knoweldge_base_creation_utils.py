"""Utilities for knowledge base creation.

This module contains the KnowledgeBaseCreation class and helper methods used by
pipeline orchestration in `knoweldge_base_creation.py`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ServiceRequestError, ServiceResponseError, HttpResponseError

    AZURE_BLOB_AVAILABLE = True
except ModuleNotFoundError:
    AZURE_BLOB_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


class KnowledgeBaseCreation(PDFTextExtractor):
    """Generates Knowledge base through structuring extracted text."""

    NOISE_PATTERNS = [
        r"\[.*?Login.*?\]\(.*?\)",
        r"Accessibility Tools",
        r"\* Invert colors.*?Letter spacing\s+100%",
        r"Search\s+Search",
        r"\[-A\].*?\[\+A\]",
        r"Menu\n",
        r"Skip to content",
        r"Copyright.*?All rights reserved.*",
        r"Best Viewed in.*?\d+ x \d+.*",
        r"\d+\n\d+\n\d+\n\d+ Visitors",
    ]

    def __init__(
        self,
        checkpoint_dir: str = "extracted_text_data/checkpoints",
        university: str = "coep",
        extraction_method: str = "azure",
        run_timestamp: Optional[str] = None,
    ):
        super().__init__(university=university, extraction_method=extraction_method)
        self.structured_documents: List[Document] = []
        self._seen_content_hashes: Set[str] = set()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.university = university
        self.extraction_method = extraction_method
        self.run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.azure_blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.azure_blob_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
        self.run_root_prefix = os.getenv("AZURE_RUNS_BLOB_PREFIX", "processing_runs")
        self.run_blob_folder = f"{self.run_root_prefix}/{self.run_timestamp}"
        self.checkpoints_blob_folder = f"{self.run_blob_folder}/checkpoints"
        self.final_processed_blob_folder = f"{self.run_blob_folder}/final_processed"
        self.tables_blob_prefix = f"{self.run_blob_folder}/extraction_processing/tables"
        self._blob_service_client = None
        self._blob_container_client = None
        logger.info("Run blob folder: %s", self.run_blob_folder)
        logger.info("Checkpoints blob folder: %s", self.checkpoints_blob_folder)
        logger.info("Final processed folder: %s", self.final_processed_blob_folder)
        logger.info("PDF extraction method: %s", self.extraction_method)
        logger.info("Checkpoint run timestamp: %s", self.run_timestamp)
        self._initialize_blob_container()

    def _initialize_blob_container(self) -> None:
        if not AZURE_BLOB_AVAILABLE:
            raise ImportError(
                "azure-storage-blob is not installed. "
                "Install it with: pip install azure-storage-blob"
            )

        if not self.azure_blob_connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. "
                "Checkpoint uploads require Azure Blob Storage configuration."
            )

        self._blob_service_client = BlobServiceClient.from_connection_string(
            self.azure_blob_connection_string
        )
        self._blob_container_client = self._blob_service_client.get_container_client(
            self.azure_blob_container_name
        )
        if not self._blob_container_client.exists():
            self._blob_container_client.create_container()
            logger.info(
                "Created Azure Blob container for checkpoints: %s",
                self.azure_blob_container_name,
            )

    def save_checkpoint(self, data: any, step_name: str, timestamp: Optional[str] = None) -> str:
        if timestamp is None:
            timestamp = self.run_timestamp

        checkpoint_file_name = f"{step_name}_{timestamp}.json"
        blob_path = f"{self.checkpoints_blob_folder}/{checkpoint_file_name}"

        try:
            if isinstance(data, list) and data and isinstance(data[0], Document):
                serializable_data = [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in data
                ]
            else:
                serializable_data = data

            payload = json.dumps(serializable_data, indent=2, ensure_ascii=False).encode("utf-8")
            self._upload_blob_bytes_with_retry(blob_path, payload)

            logger.info(
                "✅ Checkpoint uploaded to Azure Blob: %s/%s (%s items)",
                self.azure_blob_container_name,
                blob_path,
                len(data) if isinstance(data, list) else "N/A",
            )
            return blob_path
        except Exception as e:
            logger.error("Failed to save checkpoint %s: %s", step_name, e)
            raise

    def clear_tables_checkpoint_dir(self) -> None:
        tables_dir = Path(self.tables_output_dir)
        if not tables_dir.exists():
            return
        for entry in tables_dir.iterdir():
            if entry.is_file():
                entry.unlink()

    def load_checkpoint(self, checkpoint_file: str, as_documents: bool = False) -> any:
        try:
            blob_client = self._blob_container_client.get_blob_client(blob=checkpoint_file)
            payload = blob_client.download_blob().readall().decode("utf-8")
            data = json.loads(payload)

            if as_documents and isinstance(data, list):
                documents = [
                    Document(
                        page_content=item.get("page_content", ""),
                        metadata=item.get("metadata", {}),
                    )
                    for item in data
                ]
                logger.info("Loaded %s documents from checkpoint", len(documents))
                return documents

            logger.info("Loaded checkpoint from Azure Blob path %s", checkpoint_file)
            return data
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_file, e)
            raise

    def save_final_processed_documents(self, documents: List[Document]) -> str:
        """Save canonical final processed documents artifact for downstream embedding ingestion."""
        blob_path = f"{self.final_processed_blob_folder}/final_processed_documents.json"
        serializable_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

        payload = json.dumps(serializable_data, indent=2, ensure_ascii=False).encode("utf-8")
        self._upload_blob_bytes_with_retry(blob_path, payload)
        logger.info(
            "✅ Final processed document set uploaded: %s/%s (%s docs)",
            self.azure_blob_container_name,
            blob_path,
            len(documents),
        )
        return blob_path

    def _upload_blob_bytes_with_retry(self, blob_path: str, payload: bytes) -> None:
        """Upload payload bytes to Azure blob with retries for transient failures."""
        max_attempts = max(1, int(os.getenv("BLOB_UPLOAD_MAX_RETRIES", "5")))
        base_delay = max(0.5, float(os.getenv("BLOB_UPLOAD_BASE_DELAY_SECONDS", "1.5")))
        max_delay = max(base_delay, float(os.getenv("BLOB_UPLOAD_MAX_DELAY_SECONDS", "20")))

        for attempt in range(1, max_attempts + 1):
            try:
                blob_client = self._blob_container_client.get_blob_client(blob=blob_path)
                blob_client.upload_blob(payload, overwrite=True)
                return
            except Exception as e:
                retryable = isinstance(
                    e,
                    (
                        ServiceRequestError,
                        ServiceResponseError,
                        TimeoutError,
                        ConnectionError,
                    ),
                )

                if isinstance(e, HttpResponseError):
                    status_code = getattr(e, "status_code", None)
                    retryable = retryable or status_code in {408, 409, 429, 500, 502, 503, 504}

                if not retryable or attempt >= max_attempts:
                    raise

                delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.75)
                logger.warning(
                    "Blob upload transient failure for %s (attempt %d/%d): %s. Retrying in %.1fs",
                    blob_path,
                    attempt,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)

    def load_documents_from_blob_path(self, blob_path: str) -> List[Document]:
        """Load serialized documents from blob path and return LangChain Document objects."""
        blob_client = self._blob_container_client.get_blob_client(blob=blob_path)
        payload = blob_client.download_blob().readall().decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, list):
            raise ValueError(f"Expected list payload at blob path {blob_path}")

        documents = [
            Document(
                page_content=item.get("page_content", ""),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
        logger.info("Loaded %s documents from blob path %s", len(documents), blob_path)
        return documents

    def structure_documents(self, extracted_docs: List[Document]) -> List[Document]:
        if not extracted_docs:
            logger.warning("No documents to structure")
            return []

        logger.info("Structuring %s documents...", len(extracted_docs))

        structured = []
        for idx, doc in enumerate(extracted_docs):
            try:
                content = doc.page_content
                metadata = dict(doc.metadata) if doc.metadata else {}

                source = metadata.get("source", "unknown")
                page = metadata.get("page")

                metadata.update(
                    {
                        "source": source,
                        "page": str(page) if page is not None else None,
                        "university": self.university,
                    }
                )

                structured.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning("Failed to structure document %s: %s", idx, e)
                continue

        logger.info("Successfully structured %s documents", len(structured))
        return structured

    def clean_markdown_content(self, content: str) -> str:
        cleaned = content
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        lines = cleaned.split("\n")
        seen_lines = {}
        filtered_lines = []

        for line in lines:
            if not line.strip():
                filtered_lines.append(line)
                continue

            if line.strip().startswith("*") or line.strip().startswith("-"):
                line_hash = hashlib.md5(line.encode()).hexdigest()
                if line_hash in seen_lines:
                    continue
                seen_lines[line_hash] = True

            filtered_lines.append(line)

        cleaned = "\n".join(filtered_lines)
        cleaned = re.sub(r"https?://[^\s\)]{200,}", "[URL]", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def classify_document_type(self, content: str, source: str, title: str) -> str:
        content_lower = content.lower()
        source_lower = source.lower()
        title_lower = title.lower() if title else ""

        if any(
            term in content_lower or term in source_lower or term in title_lower
            for term in ["fee", "fees", "payment", "tuition"]
        ):
            return "fees"
        if any(
            term in content_lower or term in source_lower
            for term in ["admission", "apply", "eligibility", "entrance"]
        ):
            return "admissions"
        if any(term in source_lower for term in ["brochure", "flyer"]):
            return "brochure"
        if any(term in content_lower for term in ["contact", "email", "phone", "address"]):
            return "contact"
        if any(
            term in content_lower
            for term in ["program", "course", "curriculum", "b.tech", "m.tech"]
        ):
            return "programs"
        if any(term in content_lower for term in ["manual", "guide", "instruction"]):
            return "manual"
        return "general"

    def extract_metadata_from_content(self, content: str) -> dict:
        metadata = {}

        date_patterns = [
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))
        if dates:
            metadata["dates"] = list(set(dates[:5]))

        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content)
        if emails:
            metadata["emails"] = list(set(emails[:3]))

        phones = re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b", content)
        if phones:
            metadata["phones"] = list(set(phones[:3]))

        return metadata

    def deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        unique_chunks = []
        buckets = {}

        def normalize_for_similarity(text: str) -> str:
            text = re.sub(r"\s+", " ", text.lower()).strip()
            text = re.sub(r"\d+", "0", text)
            return text

        def shingle_hashes(text: str, size: int = 5) -> set:
            tokens = text.split()
            if not tokens:
                return set()
            if len(tokens) <= size:
                joined = " ".join(tokens)
                return {hashlib.md5(joined.encode("utf-8")).hexdigest()}
            hashes = set()
            for i in range(len(tokens) - size + 1):
                shingle = " ".join(tokens[i : i + size])
                hashes.add(hashlib.md5(shingle.encode("utf-8")).hexdigest())
            return hashes

        def jaccard_similarity(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        for chunk in chunks:
            normalized = normalize_for_similarity(chunk.page_content)
            content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

            if content_hash in self._seen_content_hashes:
                logger.debug("Skipping duplicate chunk from %s", chunk.metadata.get("source", "unknown"))
                continue

            shingles = shingle_hashes(normalized)
            length_bucket = max(1, len(normalized) // 200)
            bucket = buckets.setdefault(length_bucket, [])

            is_near_duplicate = False
            for candidate in bucket[-200:]:
                similarity = jaccard_similarity(shingles, candidate["shingles"])
                if similarity >= 0.95:
                    is_near_duplicate = True
                    break

            if is_near_duplicate:
                logger.debug("Skipping near-duplicate chunk from %s", chunk.metadata.get("source", "unknown"))
                continue

            self._seen_content_hashes.add(content_hash)
            bucket.append({"shingles": shingles})
            unique_chunks.append(chunk)

        logger.info("Deduplicated %s chunks to %s unique chunks", len(chunks), len(unique_chunks))
        return unique_chunks

    def website_structure_documents(self, web_data: List[dict]) -> List[Document]:
        raw = web_data or []

        if not raw:
            logger.warning("No website data available to structure")
            return []

        logger.info("Processing %s web documents...", len(raw))

        pages = []
        for idx, doc in enumerate(raw):
            try:
                raw_content = doc.get("metadata", {}).get("markdowntext", "")
                source = doc.get("metadata", {}).get("source", "unknown")
                title = doc.get("metadata", {}).get("title", "")

                cleaned_content = self.clean_markdown_content(raw_content)
                if len(cleaned_content.strip()) < 100:
                    logger.warning("Skipping document %s - too short after cleaning", idx)
                    continue

                doc_type = self.classify_document_type(cleaned_content, source, title)
                extracted_metadata = self.extract_metadata_from_content(cleaned_content)
                doc_id = idx + 5000

                pages.append(
                    Document(
                        id=doc_id,
                        page_content=cleaned_content,
                        metadata={
                            "source": source,
                            "title": title,
                            "page": title,
                            "doc_type": doc_type,
                            "source_type": "web",
                            "university": self.university,
                            **extracted_metadata,
                        },
                    )
                )
            except Exception as e:
                logger.warning("Failed to process web document %s: %s", idx, e)
                continue

        logger.info("Loaded %s pages from website data (after cleaning)", len(pages))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )

        docs = splitter.split_documents(pages)
        logger.info("Split web data into %s chunks (before deduplication)", len(docs))

        docs = self.deduplicate_chunks(docs)
        return self.structure_documents(docs)
