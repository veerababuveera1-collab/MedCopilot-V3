# ============================================================
# Clinical Research AI Copilot - Upgraded Cloud-Safe RAG App
# Author: Veera Babu
# Research only | Not for diagnosis
# ============================================================

import os
import tempfile
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

# ------------------------------------------------
# LOAD ENV + FORCE KEY
# ------------------------------------------------

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("❌ OPENAI_API_KEY missing. Add it in Streamlit Secrets or .env file.")
    st.stop()

# ------------------------------------------------
# APP CONFIG
# ------------------------------------------------

st.set_page_config(page_title="Clinical Research AI Copilot", layout="wide")

MODEL_NAME = "gpt-4o-mini"
Entrez.email = "your_email@example.com"  # change to your email

# ------------------------------------------------
# AI MODELS (FORCED AUTH)
# ------------------------------------------------

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=API_KEY
)

# ------------------------------------------------
# INGESTION
# ------------------------------------------------

def fetch_arxiv(query, n):
    docs = []
    for r in arxiv.Search(query=query, max_results=n).results():
        docs.append(Document(
            page_content=f"Title: {r.title}\nSummary: {r.summary}",
            metadata={"title": r.title, "source": r.entry_id}
        ))
    return docs


def fetch_pubmed(query, n):
    ids = Entrez.read(
        Entrez.esearch(db="pubmed", term=query, retmax=n)
    )["IdList"]

    docs = []
    for pid in ids:
        rec = Entrez.read(
            Entrez.efetch(db="pubmed", id=pid, retmode="xml")
        )
        art = rec["PubmedArticle"][0]
        title = art["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = art["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]

        docs.append(Document(
            page_content=f"Title: {title}\nAbstract: {abstract}",
            metadata={"title": title, "source": f"PubMed:{pid}"}
        ))
    return docs


def load_pdfs(files):
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
    return docs

# ------------------------------------------------
# VECTOR DB
# ------------------------------------------------

def build_vector_db(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)

# ------------------------------------------------
# RAG
# ------------------------------------------------

PROMPT = ChatPromptTemplate.from_template("""
You are a clinical research assistant.

Use only provided evidence.

Context:
{context}

Question:
{question}

Return:
• Summary
• Key findings
• Citations
""")

def build_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    def join_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | join_docs, "question": lambda x: x}
        | PROMPT
        | llm
        | StrOutputParser()
    )

# ------------------------------------------------
# UI
# ------------------------------------------------

st.title("🏥 Clinical Research AI Copilot")

with st.sidebar:
    st.header("📚 Load Knowledge")

    topic = st.text_input("Search topic", "glioblastoma treatment")

    arxiv_n = st.slider("ArXiv papers", 1, 8, 3)
    pubmed_n = st.slider("PubMed articles", 1, 8, 3)

    pdfs = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("Build Knowledge Base"):
        with st.spinner("Collecting sources..."):
            docs = []
            docs += fetch_arxiv(topic, arxiv_n)
            docs += fetch_pubmed(topic, pubmed_n)
            docs += load_pdfs(pdfs)

        if not docs:
            st.warning("No documents loaded")
            st.stop()

        with st.spinner("Building AI index..."):
            st.session_state.db = build_vector_db(docs)

        st.success(f"Indexed {len(docs)} documents!")

# ------------------------------------------------
# QA
# ------------------------------------------------

if "db" not in st.session_state:
    st.session_state.db = None

st.divider()

question = st.text_input("🔍 Ask a clinical research question")

if question and st.session_state.db:
    chain = build_chain(st.session_state.db)

    with st.spinner("Analyzing evidence..."):
        result = chain.invoke(question)

    st.subheader("📄 AI Answer")
    st.write(result)

# ------------------------------------------------
# CITATIONS
# ------------------------------------------------

if st.session_state.db and question:
    st.subheader("📚 Evidence Used")

    for doc in st.session_state.db.similarity_search(question, k=4):
        st.markdown(f"""
**📄 {doc.metadata.get('title','Document')}**  
Source: {doc.metadata.get('source')}
""")
        st.divider()

st.caption("For research purposes only – not medical advice")
