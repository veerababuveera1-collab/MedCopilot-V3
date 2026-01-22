import streamlit as st
import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ===============================
# CONFIG
# ===============================
GROK_API_KEY = "YOUR_GROK_API_KEY"
GROK_MODEL = "grok-2-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config("NETRA-RESEARCH AI", layout="wide")

st.title("🧠 NETRA-RESEARCH AI™")
st.subheader("World-Class AI Research Assistant (Powered by Grok)")

# ===============================
# LOAD EMBEDDING MODEL
# ===============================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

embed_model = load_embedding_model()

# ===============================
# VECTOR DATABASE
# ===============================
class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add_texts(self, texts):
        embeddings = embed_model.encode(texts)
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)

    def search(self, query, k=5):
        query_emb = embed_model.encode([query])
        D, I = self.index.search(np.array(query_emb).astype("float32"), k)
        return [self.texts[i] for i in I[0]]

# ===============================
# PDF LOADER
# ===============================
def load_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ===============================
# GROK AI ENGINE
# ===============================
def research_answer(query, context=""):
    url = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are NETRA-RESEARCH AI — a world-class research intelligence system.

Context:
{context}

User Question:
{query}

Provide a detailed, professional research-grade answer.
"""

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a world-class research AI"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        return f"❌ API Error: {response.text}"

    return response.json()["choices"][0]["message"]["content"]

# ===============================
# INIT VECTOR DB
# ===============================
if "vector_db" not in st.session_state:
    st.session_state.vector_db = VectorStore()

vector_db = st.session_state.vector_db

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🔬 Research Tools")
mode = st.sidebar.selectbox("Select Mode", ["AI Chat", "PDF Research"])

# ===============================
# PDF UPLOAD MODE
# ===============================
if mode == "PDF Research":
    uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Indexing document..."):
            text = load_pdf_text(uploaded_file)
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            vector_db.add_texts(chunks)
        st.sidebar.success("📄 Document indexed successfully!")

# ===============================
# QUERY INPUT
# ===============================
query = st.text_input("Enter your research question")

# ===============================
# RESEARCH BUTTON
# ===============================
if st.button("🔍 Research"):
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        if mode == "PDF Research" and len(vector_db.texts) > 0:
            docs = vector_db.search(query)
            context = "\n".join(docs)
        else:
            context = ""

        with st.spinner("NETRA AI is thinking with Grok..."):
            answer = research_answer(query, context)

        st.markdown("### 📌 Research Output")
        st.write(answer)
