# backend/utils.py
import re
import docx2txt
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- text extraction ----
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_text_from_docx(path):
    return docx2txt.process(path)


# ---- skills extraction (simple dictionary-based) ----
SKILLS_DB = [
    "python", "java", "c++", "machine learning", "deep learning", "nlp",
    "data analysis", "sql", "tensorflow", "pytorch", "scikit-learn", "docker",
    "kubernetes", "aws", "gcp", "azure", "html", "css", "javascript", "react",
    "node.js", "git", "jira", "pandas", "numpy", "matplotlib", "seaborn"
]


def extract_skills_from_text(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS_DB:
        # use word boundary to avoid partial matches
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


# ---- resume-job description matching ----
def compute_similarity_and_matches(resume_text, jd_text, top_k=20):
    """
    Computes cosine similarity and identifies matched keywords.
    
    Args:
    resume_text: text content of the resume
    jd_text: text content of the job description
    top_k: number of top keywords to extract from the JD

    Returns:
    score: cosine similarity between resume and job description
    matched_keywords_list: top keywords from JD that are present in resume
    """
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()

    # If no JD provided => score 0
    if not jd_text:
        return 0, []

    corpus = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    try:
        tfidf = vectorizer.fit_transform(corpus)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]  # between resume and JD
        score = float(sim) * 100.0
    except Exception:
        score = 0.0

    # Get top keywords from JD via TF-IDF feature scores
    try:
        jd_vec = vectorizer.transform([jd_text])
        feature_array = vectorizer.get_feature_names_out()
        tfidf_sorting = jd_vec.toarray().flatten().argsort()[::-1]
        top_n = tfidf_sorting[:top_k]
        top_keywords = [feature_array[i] for i in top_n if jd_vec[0, i] > 0][:top_k]
    except Exception:
        top_keywords = []

    # Filter keywords to those that also appear literally in resume (simple)
    resume_lower = resume_text.lower()
    matched_keywords = [k for k in top_keywords if k in resume_lower]

    return score, matched_keywords