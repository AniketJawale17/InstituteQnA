
from langchain_community.document_loaders import PyPDFLoader,UnstructuredPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from multiprocessing import Pool
from langchain_core.documents import Document

class PDFTextExtractor:
    """Utility class to extract text from PDF files."""

    def __init__(
              self, 
              pdf_path: str = "attachments/brochure.pdf",
              chunk_size: int = 2000, 
              chunk_overlap: int = 200
            ):
        
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extracted_text: list[Document] = []

 
    def extract_text_from_text_pdf(
            self
        ) -> list[str]:
            """Extract text from a PDF file and split it into pages then chunks.
            """

            path = Path(self.pdf_path)
            if not path.exists():
                raise FileNotFoundError(self.pdf_path)

            try:
                loader = PyPDFLoader(str(path))
                pages = loader.load()
            except Exception as e:
                raise RuntimeError("No PDF extraction backend available") from e
            
            # Step 2: Split pages into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            self.extracted_text = splitter.split_documents(pages)

            print(f"✅ Split {self.pdf_path} into {len(self.extracted_text)} text chunks")
            # print(chunks[0].page_content[:300])

            return self.extracted_text
    
    def extract_multiple_pdfs(
            self,
            pdf_folder_path: str
        ) -> list[str]:
            """Extract text from multiple PDF files and split them into pages then chunks.
            """
            all_chunks = []
            pdf_paths = list(Path(pdf_folder_path).glob("*.pdf"))
            # multiprocessing_pool = Pool(processes=4)
            # all_chunks = multiprocessing_pool.map(self.extract_text_from_text_pdf, pdf_paths)
            for pdf_path in pdf_paths:
                self.pdf_path = pdf_path
                chunks = self.extract_text_from_text_pdf()
                all_chunks.extend(chunks)
            return all_chunks
    
    def extract_text_from_hybrid_pdf(
            self
        ) -> list[str]:
            """Extract text from a PDF file and split it into pages then chunks.
            """

            path = Path(self.pdf_path)
            if not path.exists():
                raise FileNotFoundError(self.pdf_path)

            text_pages: list[str] = []
            try:
                loader = UnstructuredPDFLoader(str(path))
                pages = loader.load()
            except Exception as e:
                raise RuntimeError("No PDF extraction backend available") from e
            
            # Step 2: Split pages into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            chunks = splitter.split_documents(pages)

            print(f"✅ Split {self.pdf_path} into {len(chunks)} text chunks")
            # print(chunks[0].page_content[:300])

            
            return chunks



            

