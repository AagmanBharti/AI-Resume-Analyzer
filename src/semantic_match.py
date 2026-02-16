from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once (fast after first load)
model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_similarity(resume_text, job_desc):
    """
    Compute semantic similarity using embeddings.
    """

    resume_embedding = model.encode([resume_text])
    jd_embedding = model.encode([job_desc])

    score = cosine_similarity(resume_embedding, jd_embedding)[0][0]

    return round(score * 100, 2)
