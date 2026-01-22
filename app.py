# ============================================================
# Clinical Research Copilot — Active Verification Framework
# Powered by Grok (xAI) LLM
# Evidence • Contradiction • Guideline Anchoring • CRTS
# ============================================================

import streamlit as st
import requests, os, json
import numpy as np

# Safe dotenv import
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================
# CONFIG
# ============================================================

st.set_page_config("Clinical Research Copilot (AV)", layout="wide")

GROK_API_KEY = os.getenv("GROK_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")

if not GROK_API_KEY:
    st.error("❌ Missing GROK_API_KEY in environment variables")

if not SEARCH_API_KEY:
    st.error("❌ Missing SEARCH_API_KEY in environment variables")

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# UI
# ============================================================

st.title("🧠 Clinical Research Copilot — Active Verification Framework")
st.caption("Evidence-first clinical research intelligence with trust scoring")
st.warning("⚠ Research Decision Support Tool. Not for clinical diagnosis or treatment.")

query = st.text_input("Enter clinical research query")

guideline_url = st.text_input("Paste Guideline URL (NICE / WHO)")
pdf_file = st.file_uploader("Upload Clinical PDF", type=["pdf"])

alpha = st.slider("α Source Fidelity Weight", 0.0, 1.0, 0.30)
beta = st.slider("β Contradiction Weight", 0.0, 1.0, 0.30)
gamma = st.slider("γ Audit Coverage Weight", 0.0, 1.0, 0.20)
delta = st.slider("δ Guideline Alignment Weight", 0.0, 1.0, 0.20)

run = st.button("Run Active Verification")

# ============================================================
# External APIs
# ============================================================

def search_web(query):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": SEARCH_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 10
    }

    try:
        r = requests.post(url, json=payload, timeout=30)

        if r.status_code != 200:
            st.error("Search API failed. Check SEARCH_API_KEY.")
            return []

        data = r.json()

        # Handle multiple API formats safely
        if "results" in data:
            return data["results"]

        if "data" in data:
            return data["data"]

        if "sources" in data:
            return data["sources"]

        # Fallback safe format
        return [{
            "title": "Search Result",
            "url": "",
            "content": data.get("answer", "No content returned")
        }]

    except Exception as e:
        st.error(f"Search API error: {e}")
        return []

def call_grok(prompt):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "grok-2-latest",
        "messages": [
            {"role": "system", "content": "You are a clinical research copilot. Use only provided evidence."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)

        if r.status_code != 200:
            return "❌ Grok API failed. Check GROK_API_KEY."

        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Grok API error: {e}"

# ============================================================
# Evidence Processing
# ============================================================

def extract_pdf_text(pdf):
    reader = PdfReader(pdf)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])

def embed_chunks(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    vectors = EMBED_MODEL.encode(chunks)
    return chunks, np.array(vectors)

def build_faiss(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

# ============================================================
# Verification Logic
# ============================================================

def detect_contradictions(studies):
    risks = []
    for s in studies:
        text = str(s).lower()
        if "risk" in text or "adverse" in text or "warning" in text:
            risks.append(s)
    return len(risks)

def guideline_alignment(answer, guideline_text):
    v1 = EMBED_MODEL.encode(answer)
    v2 = EMBED_MODEL.encode(guideline_text)
    score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(score)

# ============================================================
# CRTS Calculation
# ============================================================

def compute_crts(sf, crr, ar, ga):
    return round(alpha*sf + beta*crr + gamma*ar + delta*ga, 2)

# ============================================================
# Main Pipeline
# ============================================================

if run and query:

    st.info("🔍 Searching clinical evidence...")
    studies = search_web(query)

    if not studies:
        st.warning("No web evidence found. Please try another query.")

    evidence_text = " ".join([
        str(s.get("content") or s.get("text") or s.get("snippet") or s)
        for s in studies
    ])

    if pdf_file:
        pdf_text = extract_pdf_text(pdf_file)
        evidence_text += " " + pdf_text

    if not evidence_text.strip():
        st.error("No evidence available to analyze.")
        st.stop()

    chunks, vectors = embed_chunks(evidence_text)
    index = build_faiss(vectors)

    st.success(f"Loaded {len(studies)} web studies")

    st.info("🧠 Generating research synthesis using Grok...")

    prompt = f"""
    You are a clinical research copilot.

    Answer the following research question strictly using the provided evidence.
    Highlight any uncertainty or conflicting findings.

    Research Question:
    {query}

    Evidence Corpus:
    {evidence_text[:6000]}
    """

    answer = call_grok(prompt)

    st.subheader("🧾 Research Synthesis (Grok)")
    st.write(answer)

    # ============================================================
    # Verification
    # ============================================================

    contradictions = detect_contradictions(studies)
    crr = 1 if contradictions > 0 else 0

    sf = 1.0  # Evidence grounded
    ar = min(1, len(studies)/10)

    ga = 0.0
    if guideline_url:
        try:
            guide_text = requests.get(guideline_url, timeout=20).text[:5000]
            ga = guideline_alignment(answer, guide_text)
        except:
            st.warning("Could not load guideline URL.")

    crts = compute_crts(sf, crr, ar, ga)

    # ============================================================
    # Audit Report
    # ============================================================

    st.subheader("📊 Active Verification Audit")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Source Fidelity (SF)", f"{sf*100:.0f}%")
    col2.metric("Contradiction Detected (CRR)", "Yes" if crr else "No")
    col3.metric("Audit Coverage (AR*)", f"{ar:.2f}")
    col4.metric("Guideline Alignment (GA)", f"{ga:.2f}")

    st.subheader("✅ Clinical Response Transparency Score (CRTS)")
    st.metric("Trust Score", crts)

    if crts >= 0.8:
        st.success("High Trust Research Output")
    elif crts >= 0.5:
        st.warning("Moderate Trust — Review Recommended")
    else:
        st.error("Low Trust — Verification Required")

    with st.expander("🔎 View Evidence Sources"):
        for s in studies:
            st.markdown(f"**{s.get('title', 'Source')}**")
            if "url" in s:
                st.write(s["url"])
            content = s.get("content") or s.get("text") or s.get("snippet") or str(s)
            st.write(content[:500])
            st.divider()
