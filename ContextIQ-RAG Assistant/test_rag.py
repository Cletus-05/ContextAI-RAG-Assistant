from rag_pipeline import get_rag_chain

rag_chain = get_rag_chain()

answer = rag_chain.invoke("What is supervised learning?")
print(answer)
