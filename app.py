# ============================================================
# Clinical Research AI Copilot – Rate Limit Proof RAG App
# Author: Veera Babu
# ============================================================

import os
import tempfile
import time
import streamlit as st
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

from Bio import Entrez
import arxiv

# ---------------- CONFIG ----------------

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("OPENAI_API_KEY missing")
    st.stop()

st.set_page_config(page_title="Clinical Research AI Copilot", layout="wide")

MODEL_NAME = "gpt-4o-mini"
VECTOR_PATH = "vector_store"
Entrez.email = "your_email@example.com"

# ---------------- AI ----------------

llm = ChatOpenAI(model=MODEL_NAME, temperature=0, openai_api_key=API_KEY)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=API_KEY
)

# ---------------- INGEST ----------------

def fetch_arxiv(q, n):
    return [
        Document(f"{r.title}\n{r.summary}", {"title": r.title, "source": r.entry_id})
        for r in arxiv.Search(query=q, max_results=n).results()
    ]

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
        docs.append(Document(f"{title}\n{abstract}", {"title": title, "source": f"PubMed:{pid}"}))
    return docs

def load_pdfs(files):
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.read())
            docs.extend(PyPDFLoader(tmp.name).load())
    return docs

# ---------------- VECTOR SAFE ----------------

def embed_safely(chunks):
    db = None
    batch_size = 8   # very safe for free tier

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

        time.sleep(3)   # cooldown

    return db


def build_or_load_vector_db(docs):
    if os.path.exists(VECTOR_PATH):
        return FAISS.load_local(VECTOR_PATH, embeddings)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    st.info(f"Embedding {len(chunks)} chunks safely...")

    db = embed_safely(chunks)
    db.save_local(VECTOR_PATH)

    return db

# ---------------- RAG ----------------

PROMPT = ChatPromptTemplate.from_template("""
You are a clinical research assistant.

Context:
{context}

Question:
{question}

Answer with summary + key findings + citations.
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

st.title("🏥 Clinical Research AI Copilot")

with st.sidebar:
    topic = st.text_input("Search topic", "glioblastoma treatment")
    arxiv_n = st.slider("ArXiv", 1, 3, 1)
    pubmed_n = st.slider("PubMed", 1, 3, 1)
    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Build Knowledge Base"):
        docs = fetch_arxiv(topic, arxiv_n) + fetch_pubmed(topic, pubmed_n) + load_pdfs(pdfs)

        if not docs:
            st.warning("No documents")
            st.stop()

        with st.spinner("Building safe vector database..."):
            st.session_state.db = build_or_load_vector_db(docs)

        st.success("Knowledge base ready!")

# ---------------- QA ----------------

if "db" not in st.session_state:
    st.session_state.db = None

question = st.text_input("Ask research question")

if question and st.session_state.db:
    chain = build_chain(st.session_state.db)
    st.write(chain.invoke(question))

st.caption("Research only")
