"""Website Scrapping functionalities"""

from langchain_community.document_loaders import WebBaseLoader
import json
import tempfile
from pathlib import Path
from typing import Any, Union, Optional
import logging
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from markdownify import markdownify as md

try:
	from azure.storage.blob import BlobServiceClient
	AZURE_BLOB_AVAILABLE = True
except ModuleNotFoundError:
	AZURE_BLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebBasedLoader:
	"""Utility class for loading and processing web-based documents."""
	
	@staticmethod
	def _make_serializable(obj: Any):
		"""Fallback serializer for non-JSON-serializable objects."""
		# LangChain Document metadata may contain datetimes or other types; coerce to string
		try:
			json.dumps(obj)
			return obj
		except Exception as e:
			return str(obj)


	@staticmethod
	def documents_to_serializable(documents):
		"""Convert a list of LangChain Document objects to plain dicts."""
		out = []
		for d in documents:
			try:
				content = d.page_content
			except Exception:
				# fallback if attribute name differs
				content = getattr(d, "content", "")

			metadata = {}
			try:
				raw_meta = getattr(d, "metadata", {}) or {}
				for k, v in raw_meta.items():
					metadata[k] = WebBasedLoader._make_serializable(v)
			except Exception:
				metadata = {"raw": WebBasedLoader._make_serializable(getattr(d, "metadata", None))}

			out.append({"page_content": content, "metadata": metadata})
		return out


	@staticmethod
	def write_json_atomic(path: Union[str, Path], data_obj, *, indent: Optional[int] = 2, ensure_ascii: bool = False):
		"""Upload JSON payload to Azure Blob using the provided path as blob name."""
		if not AZURE_BLOB_AVAILABLE:
			raise ImportError("azure-storage-blob is not installed")

		conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
		container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
		if not conn:
			raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required for blob-backed writes")

		blob_name = str(path).lstrip("/")
		payload = json.dumps(data_obj, ensure_ascii=ensure_ascii, indent=indent).encode("utf-8")
		service = BlobServiceClient.from_connection_string(conn)
		container_client = service.get_container_client(container)
		if not container_client.exists():
			container_client.create_container()
		container_client.get_blob_client(blob=blob_name).upload_blob(payload, overwrite=True)


	@staticmethod
	def load_clean_text_from_url(url: str) -> None:
		"""Load and extract text from a web page URL, then save to JSON file."""
		loader = WebBaseLoader(url)
		data = loader.load()
		# Convert documents to serializable shape and write to file
		serializable = WebBasedLoader.documents_to_serializable(data)
		WebBasedLoader.write_json_atomic("extracted_text_data/admissions_data.json", serializable)

	@staticmethod
	def load_html_markdown_from_url(url: str) -> list:
		"""Load HTML and Markdown content from a web page URL, then save to JSON file."""
		loader = WebBaseLoader(url)
		data = loader.load()
		logger.debug("Loaded %d documents from %s", len(data), url)

		# create a requests session with a small retry strategy
		session = requests.Session()
		retry_strategy = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
		session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
		session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
		session.headers.update({"User-Agent": "institute_qna/0.1"})

		# Attempt to fetch raw HTML for each document's source URL so we preserve anchor tags and other elements needed for attachment discovery.
		for d in data:
			try:
				src = getattr(d, "metadata", {}).get("source")
				# only fetch if source looks like an http URL
				if src and str(src).startswith("http"):
					try:
						r = session.get(src, timeout=20)
						r.raise_for_status()
						# keep the original extracted text in metadata for reference
						try:
							orig = getattr(d, "page_content", None)
							if orig:
								d.metadata["text_extracted"] = orig
						except Exception:
							logger.debug("Could not preserve original extracted text for a document", exc_info=True)
						# replace page_content with raw HTML so link-finder can work
						d.page_content = r.text
						d.metadata["markdowntext"] = md(r.text)
					except requests.RequestException as e:
						logger.warning("Failed to fetch raw HTML for %s: %s", src, e, exc_info=True)
			except Exception:
				# ignore documents with unexpected shapes but log at debug level
				logger.debug("Unexpected document shape while processing source", exc_info=True)
		
		# Convert documents to serializable shape and write to file
		# serializable = WebBasedLoader.documents_to_serializable(data)
		# out_path = "extracted_text_data/admissions_data.json"
		# WebBasedLoader.write_json_atomic(out_path, serializable)
		# logger.info("Wrote %d documents to %s", len(serializable), out_path)
		return data