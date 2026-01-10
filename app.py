# ================================
# MedCopilot V3 — Hybrid Hospital AI
# Evidence-based (PDF) + Global Medical AI
# ================================

import os
import streamlit as st
import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --------------------------------
# App Config
# --------------------------------
st.set_page_config(
    page_title="🧠 MedCopilot V3 — Hybrid Hospital AI",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 MedCopilot V3 — Hybrid Hospital AI")
st.caption("Evidence-based Hospital AI + Global Medical Research")
st.warning("⚠ Research support only. Not a substitute for professional medical advice.")

# --------------------------------
# Load Models (Cached)
# --------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )
    return embedder, llm

embedder, llm = load_models()

# --------------------------------
# External Medical AI (Groq)
# --------------------------------
def external_medical_ai(question):
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

    if not api_key:
        return "❌ GROQ_API_KEY not found. Please add it in Streamlit Secrets."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a medical research assistant."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ External AI Error: {str(e)}"

# --------------------------------
# Auto Medical Library Loader
# --------------------------------
def load_medical_library(folder="medical_library"):
    documents = []
    metadata = []

    if not os.path.exists(folder):
        return [], []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)

            for page_no, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text) > 200:
                    documents.append(text)
                    metadata.append(f"{file} - Page {page_no+1}")

    return documents, metadata


@st.cache_resource
def build_medical_db():
    docs, meta = load_medical_library("medical_library")

    if len(docs) == 0:
        return None, None, None

    embeddings = embedder.encode(docs)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, docs, meta


index, documents, metadata = build_medical_db()

# --------------------------------
# Sidebar Status
# --------------------------------
st.sidebar.title("🏥 MedCopilot Status")

if documents:
    st.sidebar.success(f"📚 Medical Library Loaded\n{len(documents)} pages indexed")
else:
    st.sidebar.warning("⚠ No Medical Library Found\nExternal AI Mode Enabled")

# --------------------------------
# Clinical Workspace
# --------------------------------
st.markdown("## 🔬 Clinical Research Workspace")
query = st.text_input("Ask a clinical research question:")

# --------------------------------
# Hybrid AI Engine
# --------------------------------
if query:

    # -------- Evidence-Based Hospital Mode --------
    if documents:
        q_emb = embedder.encode([query])
        D, I = index.search(np.array(q_emb), k=5)

        retrieved_docs = [documents[i] for i in I[0]]
        retrieved_sources = [metadata[i] for i in I[0]]

        context = "\n\n".join(retrieved_docs[:2])

        prompt = f"""
You are a hospital-grade medical research AI.
Answer strictly from medical evidence.

Context:
{context}

Question:
{query}

Provide professional clinical explanation.
"""

        with st.spinner("🧠 Analyzing hospital medical library..."):
            answer = llm(prompt)[0]["generated_text"]

        st.markdown("## 🩺 Evidence-Based Clinical Report")
        st.write(answer)

        st.markdown("### 📚 Evidence Sources")
        for src in retrieved_sources:
            st.info(src)

        st.success("Mode: Hospital Evidence AI")

    # -------- Global Medical AI Mode --------
    else:
        with st.spinner("🌍 Searching global medical research..."):
            answer = external_medical_ai(query)

        st.markdown("## 🌍 Global Medical Research Answer")
        st.write(answer)

        st.warning("Mode: External Medical AI (Research Mode)")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("🧠 MedCopilot V3 — Clinical Intelligence Platform for Evidence-Based Medicine")
