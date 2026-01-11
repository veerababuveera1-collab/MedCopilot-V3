import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="MedCopilot V3 — Hybrid Hospital AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- UI Header ----------------
st.markdown("""
# 🧠 MedCopilot V3 — Hybrid Hospital AI  
### Evidence-Based Hospital AI + Global Medical Research  
⚠ Research support only. Not a substitute for medical advice.
""")

# ---------------- Load Embedding Model ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------- Auto Medical Library Loader ----------------
documents = []
metadata = []

LIBRARY_PATH = "medical_library"

if os.path.exists(LIBRARY_PATH):
    for file in os.listdir(LIBRARY_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(LIBRARY_PATH, file)
            reader = PdfReader(pdf_path)

            for page_no, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text) > 300:
                    documents.append(text)
                    metadata.append(f"{file} - Page {page_no+1}")

# ---------------- Build FAISS Index ----------------
index = None

if documents:
    embeddings = embedder.encode(documents, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

# ---------------- Sidebar ----------------
st.sidebar.title("📚 MedCopilot Status")

if documents:
    st.sidebar.success("Hospital Medical Library Loaded")
    st.sidebar.write(f"Total Pages Indexed: {len(documents)}")
else:
    st.sidebar.warning("No Medical Library Found")
    st.sidebar.info("External AI Mode Enabled")

# ---------------- Clinical Workspace ----------------
st.markdown("## 🔬 Clinical Research Workspace")
query = st.text_input("Ask a clinical research question:")

# ---------------- Answer Engine ----------------
if query:

    # -------- Case 1: Hospital Evidence Mode --------
    if documents and index:

        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k=5)

        retrieved_docs = [documents[i] for i in I[0]]
        retrieved_sources = [metadata[i] for i in I[0]]

        context = "\n\n".join(retrieved_docs[:2])

        prompt = f"""
You are a hospital-grade clinical research AI.
Answer strictly from medical evidence.

Context:
{context}

Question:
{query}

Provide professional clinical explanation.
"""

        with st.spinner("🧠 Analyzing hospital medical library..."):
            external = external_research_answer(prompt)

        st.markdown("## 🩺 Clinical Intelligence Report")
        st.write(external["answer"])

        st.markdown("### 📚 Evidence Sources")
        for src in retrieved_sources:
            st.info(src)

        st.success("Mode: Hospital Evidence AI")

    # -------- Case 2: External Global AI Mode --------
    else:
        with st.spinner("🌍 Searching global medical research..."):
            external = external_research_answer(query)

        st.markdown("## 🌍 Global Medical Research Answer")
        st.write(external["answer"])

        st.warning("Mode: External Medical AI (Research Mode)")
        st.info("Upload PDFs into medical_library folder for hospital-grade evidence mode.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("🧠 MedCopilot V3 — Clinical Intelligence Platform for Evidence-Based Medicine")
