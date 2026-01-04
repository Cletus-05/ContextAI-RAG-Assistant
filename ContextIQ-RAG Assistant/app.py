import streamlit as st
from rag_pipeline import get_rag_chain

# Page config
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Study Buddy")
st.write("Ask questions from your uploaded study PDFs.")

# Load RAG chain once and cache it
@st.cache_resource
def load_rag():
    return get_rag_chain()

rag_chain = load_rag()

# User input
question = st.text_input("Enter your question:")

# Ask button
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)

        st.subheader("Answer")
        st.write(answer)


