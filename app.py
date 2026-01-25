# ============================================================
# Clinical Research AI Copilot — FREE LOCAL RAG (No API Keys)
# Author: Veera Babu
# ============================================================

import os
import tempfile
import streamlit as st
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from Bio import Entrez
import arxiv

# ---------------- CONFIG ----------------

st.set_page_config(page_title="Clinical Research AI Copilot (Free)", layout="wide")

VECTOR_PATH = "vector_store_free"
Entrez.email = "your_email@example.com"

# ---------------- FREE MODELS ----------------

llm = Ollama(model="llama3")   # or "mistral"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- INGEST ----------------

def fetch_arxiv(q, n):
    docs = []
    for r in arxiv.Search(query=q, max_results=n).results():
        docs.append(
            Document(
                page_content=f"{r.title}\n{r.summary}",
                metadata={"title": r.title, "source": r.entry_id}
            )
        )
    return docs


def fetch_pubmed(q, n):
    ids = Entrez.read(Entrez.esearch(db="pubmed", term=q, retmax=n))["IdList"]
    docs = []

    for pid in ids:
        rec = Entrez.read(Entrez.efetch(db="pubmed", id=pid, retmode="xml"))
        art = rec["PubmedArticle"][0]["MedlineCitation"]["Article"]

        if "Abstract" not in art:
            continue

        title = art["ArticleTitle"]
        abstract = " ".join(art["Abstract"]["AbstractText"])

        docs.append(
            Document(
                page_content=f"{title}\n{abstract}",
                metadata={"title": title, "source": f"PubMed:{pid}"}
            )
        )
    return docs


def load_pdfs(files):
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.read())
            docs.extend(PyPDFLoader(tmp.name).load())
    return docs

# ---------------- VECTOR STORE ----------------

def build_or_load_db(docs: List[Document]):
    if os.path.exists(VECTOR_PATH):
        return FAISS.load_local(VECTOR_PATH, embeddings)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_PATH)

    return db

# ---------------- RAG ----------------

PROMPT = ChatPromptTemplate.from_template("""
You are a clinical research assistant.

Context:
{context}

Question:
{question}

Give summary, key insights, and evidence.
""")

def build_chain(db):
    retriever = db.as_retriever(k=4)

    def join(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | join, "question": lambda x: x}
        | PROMPT
        | llm
        | StrOutputParser()
    )

# ---------------- UI ----------------

st.title("🏥 Clinical Research AI Copilot (100% Free)")

with st.sidebar:
    topic = st.text_input("Search topic", "glioblastoma treatment")

    arxiv_n = st.slider("ArXiv papers", 1, 3, 1)
    pubmed_n = st.slider("PubMed articles", 1, 3, 1)

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Build Knowledge Base"):
        docs = []
        docs += fetch_arxiv(topic, arxiv_n)
        docs += fetch_pubmed(topic, pubmed_n)
        docs += load_pdfs(pdfs)

        if not docs:
            st.warning("No documents loaded")
            st.stop()

        with st.spinner("Building local AI knowledge base..."):
            st.session_state.db = build_or_load_db(docs)

        st.success("Knowledge base ready!")

# ---------------- QA ----------------

if "db" not in st.session_state:
    st.session_state.db = None

question = st.text_input("Ask research question")

if question and st.session_state.db:
    chain = build_chain(st.session_state.db)
    st.write(chain.invoke(question))

st.caption("Runs fully free — no API keys")
