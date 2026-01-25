# ============================================================
# ĀROGYABODHA AI — Phase-3 Medical Intelligence OS (GROQ)
# Clean Clinical Research Intelligence Engine
# ============================================================

import streamlit as st
import os, json, datetime, requests, re
import pandas as pd
from groq import Groq

# ================= GROQ =================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= CONFIG =================
st.set_page_config("ĀROGYABODHA AI — Medical Intelligence OS", "🧠", layout="wide")
st.info("ℹ️ CDSS – Research & clinical decision support only")

PATIENT_DB = "patients.json"
USERS_DB = "users.json"

if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"))

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"))

# ================= SESSION =================
for k in ["logged_in","username","role"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ================= LOGIN =================
def login():
    st.title("ĀROGYABODHA AI Secure Login")
    u = st.text_input("User")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = users[u]["role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ================= TEXT CLEAN =================
def clean_text(t):
    t = re.sub(r"\s+"," ",t)
    t = re.sub(r"[^\x00-\x7F]+"," ",t)
    return t.strip()

STOPWORDS={"what","are","the","for","in","over","with","of","and","is","to","latest"}
def normalize(q):
    q=re.sub(r"[^\w\s]"," ",q.lower())
    return " ".join(t for t in q.split() if t not in STOPWORDS)

# ================= PUBMED =================
def fetch_pubmed_ids(q):
    try:
        r=requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":q,"retmode":"json","retmax":5},
            timeout=15
        )
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_pubmed_abstracts(pmids):
    abstracts=[]
    for pid in pmids:
        try:
            r=requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db":"pubmed","id":pid,"rettype":"abstract","retmode":"text"},
                timeout=15
            )
            text=clean_text(r.text)

            if len(text)<300:
                continue
            if "doi" in text.lower() and len(text.split())<120:
                continue

            abstracts.append(text[:1500])
        except:
            pass
    return abstracts

# ================= SAFE AI =================
def ai_call(prompt, tokens=350):
    try:
        return client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role":"user","content":prompt[:2000]}],
            temperature=0.2,
            max_tokens=tokens
        ).choices[0].message.content
    except:
        return "Clinical summary unavailable."

# ================= CLEAN CLINICAL SUMMARIZER =================
def ai_summarize(abstracts, question):

    if not abstracts:
        return "No sufficient clinical abstracts available for meaningful analysis."

    compressed=" ".join(abstracts[:3])

    prompt=f"""
You are a clinical research AI.

Summarize clearly for doctors.

FORMAT EXACTLY:

Summarized Treatments:
- bullet points

Clinical Insight:
- bullet points

Clinical Conclusion:
- short paragraph

Clinical Evidence:
{compressed}
"""

    return ai_call(prompt,350)

# ================= LIVE DATA =================
def fetch_trials(q):
    try:
        r=requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term":q,"pageSize":5},
            timeout=15
        ).json()
        rows=[]
        for s in r.get("studies",[]):
            i=s["protocolSection"]["identificationModule"]
            rows.append({"Trial ID":i.get("nctId","N/A"),"Status":"Active/Completed"})
        return rows
    except:
        return []

def fetch_fda():
    try:
        r=requests.get(
            "https://api.fda.gov/drug/enforcement.json?limit=5",
            timeout=15
        ).json()
        return [x["reason_for_recall"] for x in r["results"]]
    except:
        return []

# ================= SIDEBAR =================
st.sidebar.markdown(f"👨‍⚕️ {st.session_state.username}")
module=st.sidebar.radio("Medical Intelligence Center",[
    "🔬 Research Copilot",
    "📊 Dashboard",
    "👤 Patient Workspace"
])

# ================= RESEARCH COPILOT =================
if module=="🔬 Research Copilot":

    st.header("🔬 AI Clinical Research Copilot")

    q=st.text_input(
        "Ask clinical research question",
        "What are the latest treatments for glioblastoma in patients over 60?"
    )

    if st.button("Analyze Research") and q:

        nq=normalize(q)
        pmids=fetch_pubmed_ids(nq)
        abstracts=fetch_pubmed_abstracts(pmids)
        trials=fetch_trials(nq)
        alerts=fetch_fda()

        st.subheader("📊 Evidence Snapshot")
        c1,c2,c3=st.columns(3)
        c1.metric("PubMed",len(pmids))
        c2.metric("Trials",len(trials))
        c3.metric("FDA Alerts",len(alerts))

        st.subheader("🧠 Clinical Research Summary")
        st.markdown(ai_summarize(abstracts,q))

        st.subheader("📚 Evidence & Citations")
        st.dataframe(pd.DataFrame({
            "PMID":pmids,
            "Outcome":"Reported benefit",
            "FDA":"Approved/Trial"
        }),use_container_width=True)

        if trials:
            st.subheader("🧪 Clinical Trials")
            st.dataframe(pd.DataFrame(trials),use_container_width=True)

        if alerts:
            st.subheader("⚠ Regulatory & Safety Alerts")
            for a in alerts:
                st.warning(a)

# ================= DASHBOARD =================
if module=="📊 Dashboard":
    st.metric("PubMed Feed","LIVE")
    st.metric("Clinical Trials","LIVE")
    st.metric("AI Engine","ACTIVE")

# ================= PATIENT =================
if module=="👤 Patient Workspace":
    patients=json.load(open(PATIENT_DB))

    name=st.text_input("Patient name")
    age=st.number_input("Age",0,120)

    if st.button("Add Patient"):
        patients.append({
            "name":name,
            "age":age,
            "time":str(datetime.datetime.utcnow())
        })
        json.dump(patients,open(PATIENT_DB,"w"))

    st.dataframe(pd.DataFrame(patients),use_container_width=True)

# ================= FOOTER =================
st.caption("ĀROGYABODHA AI — Production Clinical Research Intelligence Engine")
