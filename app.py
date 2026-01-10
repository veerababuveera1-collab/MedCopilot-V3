import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from transformers import pipeline

from external_research import external_research_answer

# ---------------- App Config ----------------
st.set_page_config(
    page_title="MedCopilot Enterprise",
    layout="wide",
    page_icon="🧠"
)

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)
    return embedder, llm

embedder, llm = load_models()

# ---------------- Header ----------------
st.markdown("""
# 🌍 MedCopilot™ Enterprise  
## Hospital-Grade Clinical Intelligence Platform  
⚠ Clinical research only. Not a substitute for medical advice.
""")

# ---------------- Sidebar ----------------
st.sidebar.title("📄 Medical Knowledge Library")
uploaded_files = st.sidebar.file_uploader(
    "Upload Medical PDFs (optional)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- PDF Processing ----------------
documents = []
metadata = []

if uploaded_files:
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page_no, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text) > 200:
                documents.append(text)
                metadata.append(f"{pdf.name} - Page {page_no+1}")

# ---------------- Vector DB ----------------
if documents:
    embeddings = embedder.encode(documents)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

# ---------------- Research Workspace ----------------
st.markdown("## 🔬 Clinical Research Workspace")
query = st.text_input("Ask a clinical research question:")

# ---------------- Hybrid Intelligence ----------------
if query:

    # CASE 1: PDFs available → Evidence mode
    if documents:
        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k=5)

        retrieved_docs = [documents[i] for i in I[0]]
        retrieved_sources = [metadata[i] for i in I[0]]

        context = "\n\n".join(retrieved_docs[:2])

        prompt = f"""
You are a hospital-grade clinical research AI.
Answer strictly from medical context.

Context:
{context}

Question:
{query}

Provide professional medical explanation.
"""

        with st.spinner("🧠 Analyzing your medical library..."):
            answer = llm(prompt)[0]["generated_text"]

        st.markdown("## ✅ Clinical Intelligence Report")
        st.write(answer)

        st.markdown("### 📚 Evidence Sources (Your Library)")
        for src in retrieved_sources:
            st.info(src)

        st.success("Source: Your Medical Knowledge Library")

    # CASE 2: No PDFs → External research mode
    else:
        with st.spinner("🔍 Searching external medical research..."):
            external = external_research_answer(query)

        st.markdown("## ✅ Clinical Research Answer")
        st.write(external["answer"])

        st.warning("Source: External Medical Research (AI)")
        st.info("Upload your own PDFs for hospital-grade validated answers.")
