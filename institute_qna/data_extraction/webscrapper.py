"""Website Scrapping functionalities"""

from langchain_community.document_loaders import WebBaseLoader
import json
import tempfile
from pathlib import Path
from typing import Any, Union, Optional
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from markdownify import markdownify as md

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
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
			json.dump(data_obj, tmp, ensure_ascii=ensure_ascii, indent=indent)
			tmp.flush()
		Path(tmp.name).replace(path)


	@staticmethod
	def load_clean_text_from_url(url: str) -> None:
		"""Load and extract text from a web page URL, then save to JSON file."""
		loader = WebBaseLoader(url)
		data = loader.load()
		# Convert documents to serializable shape and write to file
		serializable = WebBasedLoader.documents_to_serializable(data)
		WebBasedLoader.write_json_atomic("extracted_text_data/admissions_data.json", serializable)

	@staticmethod
	def load_html_markdown_from_url(url: str) -> None:
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
		print(data)
		serializable = WebBasedLoader.documents_to_serializable(data)
		out_path = "extracted_text_data/admissions_data.json"
		WebBasedLoader.write_json_atomic(out_path, serializable)
		logger.info("Wrote %d documents to %s", len(serializable), out_path)