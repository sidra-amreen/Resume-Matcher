"""Streamlit UI:  streamlit run app.py"""
import joblib
import streamlit as st
from pypdf import PdfReader

from matcher import MODEL_PATH, ResumeMatcher

st.set_page_config(page_title="Resume ↔ JD Matcher", page_icon="📄", layout="wide")


@st.cache_resource
def get_matcher():
    corpus_path = MODEL_PATH.with_name("corpus.joblib")
    return ResumeMatcher(joblib.load(corpus_path) if corpus_path.exists() else None)


def read_upload(f):
    if f is None:
        return ""
    if f.name.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    return f.read().decode("utf-8", errors="ignore")


st.title("📄 Resume vs Job Description Matcher")
left, right = st.columns(2)
with left:
    up = st.file_uploader("Resume (PDF/TXT)", type=["pdf", "txt"])
    resume = st.text_area("…or paste resume", read_upload(up), height=300)
with right:
    jd = st.text_area("Job description", height=345)

if st.button("Match", type="primary") and resume.strip() and jd.strip():
    res = get_matcher().score(resume, jd)
    st.metric("Match score", f"{res.score}%")
    st.progress(min(int(res.score), 100))
    c1, c2, c3 = st.columns(3)
    c1.success("✅ Matched skills\n\n" + (", ".join(res.matched_skills) or "—"))
    c2.error("❌ Missing skills\n\n" + (", ".join(res.missing_skills) or "—"))
    c3.info("➕ Extra skills\n\n" + (", ".join(res.extra_skills) or "—"))
    with st.expander("Feature breakdown"):
        st.json({k: round(v, 3) for k, v in res.features.items()})
    if res.missing_skills:
        st.warning(f"Tip: if truthful, add evidence of {', '.join(res.missing_skills[:5])} to your resume.")
