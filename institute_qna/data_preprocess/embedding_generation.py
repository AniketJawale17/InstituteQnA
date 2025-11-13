from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import getpass
import os
# from dotenv import load_dotenv
# load_dotenv()


class EmbeddingsGeneration:
    """ Embedding Generation Class"""

    def __init__(self):
        self.AZURE_OPENAI_ENDPOINT="https://collegeprojectapi.openai.azure.com/"
        self.AZURE_OPENAI_API_KEY="DMhyB88GcvyTOwIRa0PYCtaAIJmdxhaOgleZ0Cypz6A3myyP04rsJQQJ99BIACF24PCXJ3w3AAABACOGH7bz"
        self.AZURE_OPENAI_API_VERSION="2024-02-01"
        self.model = "text-embedding-3-small"


    def openai_embeddings_generation(
            self,
            model: str = "text-embedding-3-small",
            text:str = ""
        ):
        """Generates OpenAI Embeddings based on the model and text for inputs

        Args:
            model (str, optional): _description_. Defaults to "text-embedding-3-small".
            text (str, optional): _description_. Defaults to "".
        """

        if not isinstance(text,list):
            text = [text]

        # OpenAI Embedding model
        embeddings = AzureOpenAIEmbeddings(
            model=model,
        )

        # In Memory Vector Store Generation
        vectorstore = InMemoryVectorStore.from_texts(
            text,
            embedding=embeddings,
        )

        return vectorstore




# if __name__ == "__main__":
#     text = "LangChain is the framework for building context-aware reasoning applications"

#     generator = EmbeddingsGeneration()
#     print(generator.openai_embeddings_generation(text = text))