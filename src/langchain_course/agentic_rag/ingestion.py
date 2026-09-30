import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATABASE_DIR = Path(__file__).resolve().parent / ".chroma"

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

def build_vectorstore() -> Chroma:
    docs = [
        document
        for url in URLS
        for document in UnstructuredLoader(
            web_url=url, chunking_strategy="basic", max_characters=1000000
        ).load()
    ]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        ),
        persist_directory=str(DATABASE_DIR),
    )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=str(DATABASE_DIR),
    embedding_function=OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ),
).as_retriever()

if __name__ == "__main__":
    build_vectorstore()
