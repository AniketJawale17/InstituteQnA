from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
from langchain_core.documents import Document
import pandas as pd

class KnowledgeBaseCreation(PDFTextExtractor):
    """Generates Knowledge base through structuring extracted text.
    """
    def __init__(self):
        super().__init__()
        self.structured_documents = []

    def structure_documents(self, extracted_docs = []) -> list:
        """Structure extracted text into a list of dictionaries with 'content' and 'metadata'."""
        structured = []
        for doc in extracted_docs:
            content = doc.page_content
            metadata = doc.metadata['source'] if 'source' in doc.metadata else {}
            page = doc.metadata['page'] if 'page' in doc.metadata else None
            page_label = doc.metadata['page_label'] if 'page_label' in doc.metadata else None
            structured.append({"content": content, "metadata": metadata, "page": page, "page_label": page_label})
        return structured

    def website_structure_documents(self, extracted_docs = []) -> list:
        """Structure extracted text from website into a list of dictionaries with 'content' and 'metadata'."""
        structured = []
        for doc in extracted_docs:
            content = doc.page_content
            metadata = doc.metadata['source'] if 'source' in doc.metadata else {}
            structured.append({"content": content, "metadata": metadata})
        return structured

def main():
    obj = KnowledgeBaseCreation()
    extracted_docs = obj.extract_multiple_pdfs("attachments/")
    structured_docs = obj.structure_documents(extracted_docs)
    return structured_docs

# if __name__ == "__main__":
#     structured_docs = main()
#     print(f"Structured {len(structured_docs)} documents for knowledge base.")
#     df = pd.DataFrame(structured_docs)
#     print(df.head())
#     df.to_csv("extracted_text_data/extracted_pdf_text.csv", index=False)