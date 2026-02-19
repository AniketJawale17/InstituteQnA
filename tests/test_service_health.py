import os

import pytest
from dotenv import load_dotenv

load_dotenv()

RUN_SERVICE_TESTS = os.getenv("RUN_SERVICE_TESTS", "false").lower() in {"1", "true", "yes"}

pytestmark = pytest.mark.skipif(
    not RUN_SERVICE_TESTS,
    reason="Set RUN_SERVICE_TESTS=true to run cloud service health checks.",
)


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        pytest.skip(f"Missing required env var: {var_name}")
    return value


def test_blob_service_is_reachable() -> None:
    from azure.storage.blob import BlobServiceClient

    connection_string = _require_env("AZURE_STORAGE_CONNECTION_STRING")
    container_name = _require_env("AZURE_STORAGE_CONTAINER_NAME")

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)

    assert container_client.exists(), f"Blob container not found: {container_name}"


def test_azure_ai_search_is_reachable() -> None:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    endpoint = _require_env("AZURE_SEARCH_ENDPOINT")
    api_key = _require_env("AZURE_SEARCH_API_KEY")
    index_name = _require_env("AZURE_SEARCH_INDEX_NAME")

    index_client = SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    index = index_client.get_index(index_name)

    assert index.name == index_name


def test_embedding_model_is_reachable() -> None:
    from langchain_openai import AzureOpenAIEmbeddings

    endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    expected_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))

    embeddings = AzureOpenAIEmbeddings(
        model=embedding_model,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    vector = embeddings.embed_query("service health check")
    assert isinstance(vector, list)
    assert len(vector) == expected_dimensions


def test_generation_model_is_reachable() -> None:
    provider = os.getenv("LLM_PROVIDER", "azure").strip().lower()

    if provider == "google":
        import google.generativeai as genai

        google_api_key = _require_env("GOOGLE_API_KEY")
        model_name = os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash")

        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Reply with exactly: OK")

        text = getattr(response, "text", "") or ""
        assert isinstance(text, str)
        assert text.strip()
        return

    from langchain_openai import AzureChatOpenAI

    chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")

    llm = AzureChatOpenAI(
        model=chat_model,
        temperature=0,
        max_tokens=16,
    )
    response = llm.invoke("Reply with exactly: OK")

    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)

    assert isinstance(content, str)
    assert content.strip()
