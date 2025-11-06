from langchain_community.document_loaders import WebBaseLoader
import json
import tempfile
from pathlib import Path
from typing import Any, Union, Optional
import requests


class WebBasedLoader:
	def _make_serializable(obj: Any):
		"""Fallback serializer for non-JSON-serializable objects."""
		# LangChain Document metadata may contain datetimes or other types; coerce to string
		try:
			json.dumps(obj)
			return obj
		except Exception:
			return str(obj)


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


	def write_json_atomic(path: Union[str, Path], data_obj, *, indent: Optional[int] = 2, ensure_ascii: bool = False):
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
			json.dump(data_obj, tmp, ensure_ascii=ensure_ascii, indent=indent)
			tmp.flush()
		Path(tmp.name).replace(path)


	def load_clean_text_from_url(url: str) -> None:
		"""Load and extract text from a web page URL, then save to JSON file."""
		loader = WebBaseLoader(url)
		data = loader.load()
		# Convert documents to serializable shape and write to file
		serializable = WebBasedLoader.documents_to_serializable(data)
		WebBasedLoader.write_json_atomic("extracted_text_data/admissions_data.json", serializable)

	def load_html_markdown_from_url(url: str) -> None:
		"""Load HTML and Markdown content from a web page URL, then save to JSON file."""
		loader = WebBaseLoader(url, mode="html+markdown")
		data = loader.load()
		# Convert documents to serializable shape and write to file
		serializable = WebBasedLoader.documents_to_serializable(data)
		WebBasedLoader.write_json_atomic("extracted_text_data/admissions_data.json", serializable)
