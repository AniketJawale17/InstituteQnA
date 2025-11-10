
import logging
from unittest import loader
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFTextExtractor:
    """Utility class to extract text from PDF files."""

    @staticmethod
    def extract_text_from_pdf(
            pdf_path: str, 
            chunk_size: int = 2000, 
            chunk_overlap: int = 200
        ) -> list[str]:
            """Extract text from a PDF file and split it into pages then chunks.
            """

            path = Path(pdf_path)
            if not path.exists():
                raise FileNotFoundError(pdf_path)
                
            text_pages: list[str] = []
            try:
                loader = PyPDFLoader(str(path))
                pages = loader.load()
            except Exception as e:
                raise RuntimeError("No PDF extraction backend available") from e
            
            # Step 2: Split pages into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            chunks = splitter.split_documents(pages)

            print(f"✅ Split {pdf_path} into {len(chunks)} text chunks")
            # print(chunks[0].page_content[:300])

            
            return chunks



            

