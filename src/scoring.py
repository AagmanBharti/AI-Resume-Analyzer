import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from src.semantic_match import semantic_similarity

def hybrid_match_score(resume_text, job_desc, vectorizer):
    """
    Combine TF-IDF and Semantic similarity
    """

    tfidf_score = match_score(resume_text, job_desc, vectorizer)
    semantic_score = semantic_similarity(resume_text, job_desc)

    # Weighted fusion
    final_score = (0.6 * tfidf_score) + (0.4 * semantic_score)

    return round(final_score, 2), tfidf_score, semantic_score


def match_score(resume_text, job_desc, vectorizer):
    resume_vec = vectorizer.transform([resume_text])
    jd_vec = vectorizer.transform([job_desc])

    score = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(score * 100, 2)


def resume_quality(match_score, skill_count, text):

    # Safety check (prevents crashes)
    if not isinstance(text, str):
        text = str(text)

    word_count = len(text.split())

    # Weighted ATS scoring
    quality_score = (
    (match_score * 0.5) +
    (min(skill_count, 20) * 1.5) +
    (min(word_count, 800) / 10)
)


    return round(quality_score, 2)

def section_check(text):
    sections = {
        "Education": bool(re.search(r'\b(education|academic|qualification)\b', text)),
        "Experience": bool(re.search(r'\b(experience|employment|work history)\b', text)),
        "Projects": bool(re.search(r'\b(project|projects)\b', text)),
        "Skills": bool(re.search(r'\b(skill|skills|technical skills)\b', text)),
        "Certifications": bool(re.search(r'\b(certification|certificate)\b', text)),
    }

    return sections
