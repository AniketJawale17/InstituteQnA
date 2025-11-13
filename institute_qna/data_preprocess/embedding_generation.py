"""This module contains the Embeddings Generation class and methods"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import getpass
import os

class EmbeddingsGeneration:
    """ Embedding Generation Class"""

    def __init__(self):
        self.AZURE_OPENAI_ENDPOINT="https://collegeprojectapi.openai.azure.com/"
        self.AZURE_OPENAI_API_KEY="DMhyB88GcvyTOwIRa0PYCtaAIJmdxhaOgleZ0Cypz6A3myyP04rsJQQJ99BIACF24PCXJ3w3AAABACOGH7bz"
        self.AZURE_OPENAI_API_VERSION="2024-02-01"
        self.model = "text-embedding-3-small"


    def openai_embeddings_generation(
            self,
            docs,
            model: str = "text-embedding-3-small",
        ):
        """Generates OpenAI Embeddings based on the model and text for inputs

        Args:
            model (str, optional): _description_. Defaults to "text-embedding-3-small".
            text (str, optional): _description_. Defaults to "".
        """

        if not isinstance(docs,list):
            docs = [docs]

        # OpenAI Embedding model
        embeddings = AzureOpenAIEmbeddings(
            model=model,
        )


        self.vector_store = Chroma(
            collection_name="UG_admission_data",
            embedding_function=embeddings,
            persist_directory="./ug_admission_data",  # Where to save data locally, remove if not necessary
        )
        self.vector_store.add_documents(documents=docs)

        return self.vector_store
    
    def add_documents(self, docs: list):
        """Add documents to the existing vector store."""
        if not hasattr(self, 'vector_store'):
            raise ValueError("Vector store not initialized. Please run openai_embeddings_generation first.")
        
        self.vector_store.add_documents(documents=docs)
        return self.vector_store



# if __name__ == "__main__":
#     text = "LangChain is the framework for building context-aware reasoning applications"

#     generator = EmbeddingsGeneration()
#     print(generator.openai_embeddings_generation(text = text))