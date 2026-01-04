"""
ingest.py

Purpose:
- Load PDFs from data/
- Split text into chunks
- Create embeddings (Hugging Face)
- Store embeddings in FAISS (local vector DB)

Run this file ONCE whenever new PDFs are added.
"""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def ingest_documents():
    documents = []

    # 1️⃣ Load all PDFs
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No PDF documents found in data/ folder")

    # 2️⃣ Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # 3️⃣ Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4️⃣ Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    print("✅ Ingestion complete. Vector store saved.")


if __name__ == "__main__":
    ingest_documents()
