import streamlit as st
from document_loader import load_pdf_text
from vector_store import VectorStore
from research_engine import research_answer
from config import EMBEDDING_MODEL

st.set_page_config("NETRA-RESEARCH AI", layout="wide")

st.title("🧠 NETRA-RESEARCH AI™")
st.subheader("World-Class AI Research Assistant (Powered by Grok)")

vector_db = VectorStore(EMBEDDING_MODEL)

# Sidebar
st.sidebar.title("Research Tools")
mode = st.sidebar.selectbox("Select Mode", ["AI Chat", "PDF Research"])

# PDF Upload
if mode == "PDF Research":
    uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])
    if uploaded_file:
        text = load_pdf_text(uploaded_file)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        vector_db.add_texts(chunks)
        st.sidebar.success("📄 Document indexed successfully!")

# Chat Interface
query = st.text_input("Enter your research question")

if st.button("Research"):
    if mode == "PDF Research":
        docs = vector_db.search(query)
        context = "\n".join(docs)
    else:
        context = ""

    with st.spinner("NETRA AI is thinking with Grok..."):
        answer = research_answer(query, context)

    st.markdown("### 📌 Research Output")
    st.write(answer)
