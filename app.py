# ============================================================
# Clinical Research AI Copilot - Streamlit RAG (LangChain 0.2+)
# Author: Veera Babu
# Research only | Not for diagnosis
# ============================================================

import os
import tempfile
import streamlit as st
from typing import List

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
# CONFIG
# ------------------------------------------------

st.set_page_config("Clinical Research AI Copilot", layout="wide")
MODEL_NAME = "gpt-4o-mini"

Entrez.email = "your_email@example.com"  # REQUIRED by PubMed

# ------------------------------------------------
# LLM
# ------------------------------------------------

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
embeddings = OpenAIEmbeddings()

# ------------------------------------------------
# INGEST SOURCES
# ------------------------------------------------

def fetch_arxiv(query, max_results):
    docs = []
    for r in arxiv.Search(query=query, max_results=max_results).results():
        docs.append(Document(
            page_content=f"{r.title}\n{r.summary}",
            metadata={"title": r.title, "source": r.entry_id}
        ))
    return docs


def fetch_pubmed(query, max_results):
    ids = Entrez.read(Entrez.esearch(db="pubmed", term=query, retmax=max_results))["IdList"]
    docs = []

    for pid in ids:
        article = Entrez.read(Entrez.efetch(db="pubmed", id=pid, retmode="xml"))
        art = article["PubmedArticle"][0]
        title = art["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = art["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]

        docs.append(Document(
            page_content=f"{title}\n{abstract}",
            metadata={"title": title, "source": f"PubMed:{pid}"}
        ))
    return docs


def load_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(file.read())
            loader = PyPDFLoader(f.name)
            docs.extend(loader.load())
    return docs

# ------------------------------------------------
# VECTOR DB
# ------------------------------------------------

def build_db(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# ------------------------------------------------
# RAG CHAIN
# ------------------------------------------------

PROMPT = ChatPromptTemplate.from_template("""
You are a clinical research assistant.

Answer only using provided evidence.

Context:
{context}

Question:
{question}

Return:
• Summary
• Comparative insights
• Citations (titles)
""")

def create_chain(db):
    retriever = db.as_retriever(k=4)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "question": lambda x: x}
        | PROMPT
        | llm
        | StrOutputParser()
    )

# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------

st.title("🏥 Clinical Research AI Copilot")

with st.sidebar:
    st.header("📚 Data Sources")

    topic = st.text_input("Search topic", "glioblastoma treatment")

    arxiv_n = st.slider("ArXiv papers", 1, 10, 4)
    pubmed_n = st.slider("PubMed articles", 1, 10, 4)

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Build Knowledge Base"):
        with st.spinner("Collecting research..."):
            docs = []
            docs += fetch_arxiv(topic, arxiv_n)
            docs += fetch_pubmed(topic, pubmed_n)
            docs += load_pdfs(pdfs)

        with st.spinner("Indexing documents..."):
            st.session_state.db = build_db(docs)

        st.success(f"Indexed {len(docs)} sources!")

if "db" not in st.session_state:
    st.session_state.db = None

st.divider()

question = st.text_input("🔍 Ask clinical research question:")

if question and st.session_state.db:
    chain = create_chain(st.session_state.db)

    with st.spinner("Analyzing evidence..."):
        answer = chain.invoke(question)

    st.subheader("📄 Evidence-based Answer")
    st.write(answer)

# ------------------------------------------------
# CITATION CARDS
# ------------------------------------------------

if st.session_state.db:
    st.subheader("📚 Sources Used")

    for doc in st.session_state.db.similarity_search(question, k=4):
        with st.container():
            st.markdown(f"""
            **📄 {doc.metadata.get('title','Document')}**  
            Source: {doc.metadata.get('source')}
            """)
            st.divider()

st.caption("Research only • Not for medical diagnosis")
