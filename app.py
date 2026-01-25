# ============================================================
# 🏥 Clinical Research AI Copilot – FINAL APP
# Research & education only – Not for diagnosis
# Author: Veera Babu
# ============================================================

import os
import streamlit as st
import tempfile
from datetime import datetime
from typing import List

from Bio import Entrez
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

import pandas as pd
import plotly.express as px

# ------------------ CONFIG ------------------

st.set_page_config("Clinical AI Copilot", layout="wide")

Entrez.email = "research@example.com"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

LOG_FILE = "audit_log.csv"

# ------------------ UTILITIES ------------------

def log_action(action, query):
    row = {
        "time": datetime.now(),
        "action": action,
        "query": query
    }
    df = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def chunk_docs(texts: List[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for t in texts:
        docs.extend(splitter.create_documents([t]))
    return docs

# ------------------ INGESTION ------------------

def ingest_pubmed(query, limit=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=limit)
    ids = Entrez.read(handle)["IdList"]

    texts = []

    for pid in ids:
        fetch = Entrez.efetch(db="pubmed", id=pid, rettype="abstract", retmode="text")
        texts.append(fetch.read())

    return texts, ids

def ingest_pdfs(files):
    texts = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.read())
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            for p in pages:
                texts.append(p.page_content)
    return texts

# ------------------ VECTOR BUILD ------------------

def build_index(texts):
    docs = chunk_docs(texts)
    return FAISS.from_documents(docs, embeddings)

# ------------------ SEARCH ------------------

def search_evidence(query):
    db = st.session_state.vector_db
    return db.similarity_search(query, k=5)

# ------------------ AI TASKS ------------------

def summarize_docs(docs):
    content = "\n".join([d.page_content for d in docs])
    prompt = f"Summarize key clinical findings:\n{content}"
    return llm.invoke(prompt).content

def compare_outcomes(docs):
    content = "\n".join([d.page_content for d in docs])
    prompt = f"Compare treatment outcomes (survival, response):\n{content}"
    return llm.invoke(prompt).content

def score_evidence(docs):
    prompt = "Rate evidence strength from 1–5 based on methodology."
    text = "\n".join([d.page_content for d in docs])
    return llm.invoke(prompt + text).content

# ------------------ UI ------------------

st.title("🧠 Clinical Research AI Copilot")

tabs = st.tabs([
    "📥 Ingestion",
    "🔍 Research",
    "📊 Analytics",
    "📁 Audit Log"
])

# =====================================================
# INGESTION
# =====================================================

with tabs[0]:
    st.subheader("Unstructured Literature Ingestion")

    pub_query = st.text_input("Search PubMed")
    pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Ingest Knowledge"):
        texts = []

        if pub_query:
            pm_texts, pm_ids = ingest_pubmed(pub_query)
            texts.extend(pm_texts)
            st.success(f"Ingested {len(pm_ids)} PubMed articles")

        if pdfs:
            pdf_texts = ingest_pdfs(pdfs)
            texts.extend(pdf_texts)
            st.success(f"Ingested {len(pdfs)} PDFs")

        if texts:
            st.session_state.vector_db = build_index(texts)
            st.success("Semantic Index Built!")

# =====================================================
# RESEARCH
# =====================================================

with tabs[1]:
    st.subheader("Clinical Query Understanding")

    query = st.text_input("Ask clinical research question")

    if st.button("Search Evidence"):
        log_action("SEARCH", query)

        results = search_evidence(query)

        st.markdown("### 🔎 Retrieved Evidence")
        for i, r in enumerate(results):
            st.write(f"**Doc {i+1}:**", r.page_content[:500])

        if st.button("🧾 Summarize"):
            st.markdown(summarize_docs(results))

        if st.button("📈 Compare Outcomes"):
            st.markdown(compare_outcomes(results))

        if st.button("⭐ Evidence Strength"):
            st.markdown(score_evidence(results))

# =====================================================
# ANALYTICS DASHBOARD
# =====================================================

with tabs[2]:
    st.subheader("Real-Time Research Analytics")

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)

        fig = px.histogram(df, x="action", title="Query Activity")
        st.plotly_chart(fig)

        st.dataframe(df.tail(20))

# =====================================================
# AUDIT & COMPLIANCE
# =====================================================

with tabs[3]:
    st.subheader("Audit Log")

    if os.path.exists(LOG_FILE):
        st.dataframe(pd.read_csv(LOG_FILE))
    else:
        st.info("No logs yet")

# =====================================================
# FOOTER
# =====================================================

st.caption("Clinical AI Copilot — Evidence-based research assistant")
