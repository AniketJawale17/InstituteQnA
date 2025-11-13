"""Knowledge Base Creation Module.
This module structures extracted text from PDFs and websites into a format suitable for knowledge base creation."""

from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from pathlib import Path
import pandas as pd
from time import sleep
from dotenv import load_dotenv
load_dotenv()

class KnowledgeBaseCreation(PDFTextExtractor):
    """Generates Knowledge base through structuring extracted text.
    """
    def __init__(self):
        super().__init__()
        self.structured_documents = []

    def structure_documents(self, extracted_docs = []) -> list:
        """Structure extracted text into a list of dictionaries with 'content' and 'metadata'."""
        structured = []
        for _, doc in enumerate(extracted_docs):
            content = doc.page_content
            source = doc.metadata['source'] if 'source' in doc.metadata else {}
            page = doc.metadata['page'] if 'page' in doc.metadata else None
            page_label = doc.metadata['page_label'] if 'page_label' in doc.metadata else None
            structured.append(Document(id = _, page_content=content, metadata={"source": source, "page": str(page), "page_label": page_label}))
        return structured

    def website_structure_documents(self, webdata_file_name: str) -> list:
        """Structure extracted text from website into a list of dictionaries with 'content' and 'metadata'."""
        json_path = Path(webdata_file_name)

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        pages = []
        for _, doc in enumerate(raw):
            content = doc['metadata']['markdowntext']
            source = doc['metadata']['source'] if 'metadata' in doc and 'source' in doc['metadata'] else {}
            title = doc['metadata']['title'] if 'metadata' in doc and 'title' in doc['metadata'] else None
            i = _ +5000
            pages.append(Document(id = i, page_content=content, metadata={"source": source, "page": title, "page_label": 1000}))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        # The final output variable is `docs` for downstream steps
        docs = splitter.split_documents(pages)

        structured = self.structure_documents(docs)
        return structured

def main():
    obj = KnowledgeBaseCreation()
    # extracted_docs = obj.extract_multiple_pdfs("attachments/")
    # structured_docs = obj.structure_documents(extracted_docs)
    web_structured_docs = obj.website_structure_documents("extracted_text_data/admissions_data.json")
    # return  web_structured_docs + structured_docs
    return web_structured_docs

if __name__ == "__main__":
    structured_docs = main()

    print("_________________________________________________________")
    print("Length of structured docs:", len(structured_docs))
    print("_________________________________________________________")

    from institute_qna.data_preprocess.embedding_generation import EmbeddingsGeneration
    embed_gen =  EmbeddingsGeneration() 

    k = 70  # Number of documents to process in each batch
    if len(structured_docs) < k:
        k = len(structured_docs)
    vector_store = embed_gen.openai_embeddings_generation(
            docs = structured_docs[:k]
    )
    sleep(60)

    
    if len(structured_docs) > k:
        print("Adding more documents to the vector store...")
        for i in range(1, (len(structured_docs) - k)//k):
            new_docs = structured_docs[i*k : (i+1)*k]
            embed_gen.add_documents(new_docs)
            sleep(60)
            if i >4:
                break


    print(f"Structured {len(structured_docs)} documents for knowledge base.")
    df = pd.DataFrame(structured_docs)
    print(df.head())
    df.to_csv("extracted_text_data/extracted_pdf_text.csv", index=False)
