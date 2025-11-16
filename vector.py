from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

class VectorEmbeddings():
    def __init__(self, html_string,  embedding_model = "text-embedding-3-small", db_location = "vector_db", search_kwargs = 5):
        self.embeddings = OpenAIEmbeddings(model = embedding_model)
        self.html_string = html_string
        self.db_location = db_location
        self.add_documents = not os.path.exists(self.db_location)
        self.search_kwargs = search_kwargs

    def vectorize(self):
        retriever = None
        try:
            vector_store = None
            if self.add_documents:
                vector_store = self.create_docs()
            else:
                vector_store = self.retrieve_docs()

            retriever = vector_store.as_retriever(
                search_kwargs = {"k":self.search_kwargs}
            )           
        except Exception as err:
            print(f"[VectorEmbeddings] {err}")
        finally:
            return retriever

    def create_docs(self):
        splitter = RecursiveCharacterTextSplitter.from_language(language = "html", chunk_size = 200)

        docs = splitter.create_documents(self.html_string)

        vector_store = Chroma.from_documents(
            documents = docs,
            embedding = self.embeddings,
            persist_directory = self.db_location
        )

        return vector_store

    def retrieve_docs(self):
        vector_store = Chroma(
            persist_directory = self.db_location,
            embedding_function = self.embeddings
        )

        return vector_store