"""
rag_pipeline.py

Purpose:
- Load FAISS vector store
- Create retriever
- Build modern Runnable-based RAG chain
"""

from transformers import pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


VECTORSTORE_DIR = "vectorstore"


def get_rag_chain():
    # 1️⃣ Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2️⃣ Load FAISS index
    vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


    # 3️⃣ Load Hugging Face LLM
    hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    max_length=256
)


    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 4️⃣ Strict RAG prompt
    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.

Answer the question ONLY using the context provided below.
If the answer is NOT present in the context, say exactly:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    # 5️⃣ Build modern RAG chain
    rag_chain = (
        RunnablePassthrough()
        | {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
        )


    return rag_chain
