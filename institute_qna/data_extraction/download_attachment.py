#!/usr/bin/env python3
"""Download linked attachments found in admissions_data.json.

Scans the saved documents' HTML (`page_content`) for anchor tags linking to files
(pdf, docx, xls, images, zip, etc.), resolves relative URLs using the document's
`source` metadata, downloads files into `CollegeProject/attachments/`, and adds
an `attachments` list to each document's metadata in the JSON.

Run: python download_attachments.py
"""

#Libriries to import
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import re
from logging import getLogger
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os
from typing import List, Dict, Any
from institute_qna.logging_config import configure_logging

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_BLOB_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    AZURE_BLOB_AVAILABLE = False

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOGGER_LEVEL","INFO"))
logger.addHandler(logging.StreamHandler())
configure_logging(level=os.getenv("LOGGER_LEVEL","INFO"), log_file=None)


@dataclass
class Config:

    # file extensions we consider downloadable
    EXT_RE = re.compile(r"\.(pdf|docx|doc|xls|xlsx|csv|pptx|zip|tar.gz|gz|png|jpe?g|gif)$", re.I)


class AttachmentDownloader:
    """Utility class to find and download attachments from HTML content."""

    @staticmethod
    def find_download_links(html: str):
        """Find candidate downloadable URLs in HTML.

        Looks for <a href>, <iframe src>, <embed src> and <object data>.
        Returns raw href/src strings (may be relative).
        """
        soup = BeautifulSoup(html, "html.parser")
        links = []

        # anchor tags
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # skip javascript/mailto/etc
            if href.startswith("javascript:") or href.startswith("mailto:"):
                continue
            if a.has_attr("download") or Config.EXT_RE.search(href):
                links.append(href)

        # iframe/embed/object tags may contain documents
        for tag in ("iframe", "embed"):
            for t in soup.find_all(tag, src=True):
                src = t["src"].strip()
                if not src.startswith("javascript:"):
                    links.append(src)

        for obj in soup.find_all("object", data=True):
            data = obj["data"].strip()
            if data:
                links.append(data)

        # deduplicate while preserving order
        seen = set()
        out = []
        for l in links:
            if l not in seen:
                seen.add(l)
                out.append(l)
        return out

    @staticmethod
    def safe_filename_from_url(url: str):
        p = urlparse(url)
        name = Path(p.path).name
        if not name:
            name = p.netloc.replace(".", "_")
        return name


    @staticmethod
    def _filename_from_content_disposition(cd: str):
        # simple parser for filename from content-disposition
        if not cd:
            return None
        # look for filename="..." or filename=...
        m = re.search(r'filename\*?=([^;]+)', cd)
        if not m:
            return None
        filename = m.group(1).strip()
        # remove encoding markers and surrounding quotes
        filename = filename.split("''")[-1].strip().strip('"')
        return filename

    @staticmethod
    def download_file(url: str, filename_hint: str, session: requests.Session):
        headers = {"User-Agent": "my-scraper/0.1 (+https://example.com)"}
        try:
            # handle protocol-relative URLs
            if url.startswith("//"):
                url = "https:" + url

            r = session.get(url, stream=True, headers=headers, timeout=30, allow_redirects=True)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None

        # If destination filename has no extension, try to get a better name from
        # Content-Disposition or the final URL after redirects
        final_url = r.url
        resolved_filename = filename_hint
        if not urlparse(resolved_filename).path.lower().endswith(tuple([".pdf", ".docx", ".doc", ".xls", ".xlsx", ".csv", ".pptx", ".zip", ".tar.gz", ".gz", ".png", ".jpg", ".jpeg", ".gif"])):
            # try content-disposition
            cd = r.headers.get("content-disposition")
            filename = AttachmentDownloader._filename_from_content_disposition(cd) if cd else None
            if not filename:
                filename = AttachmentDownloader.safe_filename_from_url(final_url)
            if filename:
                resolved_filename = filename

        file_bytes = b"".join(chunk for chunk in r.iter_content(chunk_size=8192) if chunk)

        if not (AZURE_BLOB_AVAILABLE and os.getenv("AZURE_STORAGE_CONNECTION_STRING")):
            raise ValueError(
                "Azure Blob Storage is required for attachment downloads. "
                "Set AZURE_STORAGE_CONNECTION_STRING and install azure-storage-blob."
            )

        container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
        attachment_prefix = os.getenv("AZURE_ATTACHMENTS_BLOB_PREFIX", "attachments")
        blob_name = f"{attachment_prefix}/{resolved_filename}"

        service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        container_client = service.get_container_client(container)
        if not container_client.exists():
            container_client.create_container()

        container_client.get_blob_client(blob=blob_name).upload_blob(file_bytes, overwrite=True)
        return f"azure-blob://{container}/{blob_name}"

    @staticmethod
    def augment_documents_with_attachment_links(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach resolved downloadable links to each document metadata in-memory.

        This method does not write any local files and is intended for blob-first pipelines.
        """
        for idx, doc in enumerate(data):
            html = doc.get("page_content", "")
            source = doc.get("metadata", {}).get("source")
            links = AttachmentDownloader.find_download_links(html)

            resolved_links = []
            for href in links:
                if source:
                    resolved_links.append(urljoin(source, href))
                else:
                    resolved_links.append(href)

            if resolved_links:
                meta = doc.setdefault("metadata", {})
                meta["attachments"] = resolved_links
                logger.info(
                    "Doc %d: attached %d downloadable links (showing up to 5): %s",
                    idx,
                    len(resolved_links),
                    resolved_links[:5],
                )

        return data




    def download_attachments_from_documents(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Download linked attachments directly to Azure Blob and annotate docs.

        Keeps original source URLs in `metadata.attachments` and writes uploaded blob
        URIs to `metadata.attachment_blobs`.
        """

        if not (AZURE_BLOB_AVAILABLE and os.getenv("AZURE_STORAGE_CONNECTION_STRING")):
            raise ValueError(
                "Azure Blob Storage is required for attachment downloads. "
                "Set AZURE_STORAGE_CONNECTION_STRING and install azure-storage-blob."
            )

        session = requests.Session()
        for idx, doc in enumerate(data):
            html = doc.get("page_content", "")
            source = doc.get("metadata", {}).get("source")
            links = AttachmentDownloader.find_download_links(html)

            if not links:
                continue

            resolved_urls: List[str] = []
            uploaded_blob_uris: List[str] = []
            for href in links:
                url = urljoin(source, href) if source else href
                resolved_urls.append(url)

                filename_hint = AttachmentDownloader.safe_filename_from_url(url)
                logger.info("Uploading attachment to blob from URL: %s", url)
                uploaded = AttachmentDownloader.download_file(url, filename_hint, session)
                if uploaded:
                    uploaded_blob_uris.append(uploaded)

            if resolved_urls:
                meta = doc.setdefault("metadata", {})
                meta["attachments"] = resolved_urls
                meta["attachment_blobs"] = uploaded_blob_uris
                logger.info(
                    "Doc %d: %d attachment URLs, %d uploaded to blob (showing up to 5): %s",
                    idx,
                    len(resolved_urls),
                    len(uploaded_blob_uris),
                    uploaded_blob_uris[:5],
                )

        return data

    def download_all_attachments(self,input_json_path):
        """Main method to download attachments from admissions_data.json.

        Returns updated records in-memory and does not rewrite input JSON locally.
        """


        if not (AZURE_BLOB_AVAILABLE and os.getenv("AZURE_STORAGE_CONNECTION_STRING")):
            raise ValueError(
                "Azure Blob Storage is required for attachment downloads. "
                "Set AZURE_STORAGE_CONNECTION_STRING and install azure-storage-blob."
            )
        logger.info("Loading JSON from: %s", input_json_path)
        if not input_json_path.exists():
            logger.warning("admissions_data.json not found at: %s", input_json_path)
            return []
        data = json.loads(input_json_path.read_text(encoding="utf-8"))
        return self.download_attachments_from_documents(data)



# json_path = Path(__file__).resolve().parent.parent.parent / "extracted_text_data" / "admissions_data.json"

# AttachmentDownloader().download_all_attachments(json_path)



